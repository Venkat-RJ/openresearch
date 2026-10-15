#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
using namespace std; using ll=long long;
struct Solver {
 int n,m,N; vector<int> jid,r,d,w,p,mac,own,pre,suc,last,base,tail; vector<vector<int>> nodes;
 mt19937 rng; chrono::steady_clock::time_point deadline; long long evals=0,iters=0; int bestiter=0;
 vector<int> st,ind,ms,mp,q; vector<vector<int>> best; ll bv=LLONG_MAX; vector<int> bst;
 double U(){return (rng()+0.5)/4294967296.0;}
 bool timeup(){return chrono::steady_clock::now()>=deadline;}
 double decode(const vector<vector<int>>& o,ll &val, bool keep=false){
  ++evals; fill(ms.begin(),ms.end(),-1); fill(mp.begin(),mp.end(),-1); st=base;
  for(int v=0;v<N;v++) ind[v]=(pre[v]>=0);
  for(auto &a:o) for(int k=1;k<(int)a.size();k++){ms[a[k-1]]=a[k];mp[a[k]]=a[k-1];ind[a[k]]++;}
  int end=0; for(int v=0;v<N;v++)if(!ind[v])q[end++]=v;
  for(int h=0;h<end;h++){int v=q[h],t=st[v]+p[v]; for(int u:{suc[v],ms[v]})if(u>=0){st[u]=max(st[u],t);if(!--ind[u])q[end++]=u;}}
  if(end!=N){val=LLONG_MAX;return 1e100;}
  val=0;ll flow=0;for(int j=0;j<n;j++){int c=st[last[j]]+p[last[j]];val+=(ll)w[j]*max(0,c-d[j]);flow+=(ll)w[j]*c;}
  if(keep&&val<bv){bv=val;best=o;bst=st;bestiter=iters;}
  return (double)val+0.000001*flow;
 }
 vector<vector<int>> dispatch(int run){
  vector<vector<int>> o(m);vector<int> k(n),ready=r,mr(m);double avg=accumulate(p.begin(),p.end(),0.0)/N;
  double K=run==0?2.0:exp(log(0.3)+U()*log(20.0));double noise=run<2?0:0.1+U()*0.9;
  for(int step=0;step<N;step++){
   int pivot=-1,finish=INT_MAX;vector<int> est(n),v(n,-1);
   for(int j=0;j<n;j++)if(k[j]<m){v[j]=nodes[j][k[j]];est[j]=max(ready[j],mr[mac[v[j]]]);if(est[j]+p[v[j]]<finish){finish=est[j]+p[v[j]];pivot=j;}}
   int machine=mac[v[pivot]],chosen=pivot;double bs=-1e100;
   for(int j=0;j<n;j++)if(v[j]>=0&&mac[v[j]]==machine&&est[j]<finish){int a=v[j];double score=log((double)max(w[j],1)/p[a])-max(0,d[j]-est[j]-tail[a])/(K*avg);score+=noise*(-log(-log(U())));if(score>bs){bs=score;chosen=j;}}
   int a=v[chosen];o[machine].push_back(a);ready[chosen]=est[chosen]+p[a];mr[machine]=ready[chosen];k[chosen]++;
  }return o;
 }
 void move(vector<int>&a,int k,int l){int v=a[k];a.erase(a.begin()+k);a.insert(a.begin()+l,v);}
 void run(double secs,int seed){
  rng.seed(seed);deadline=chrono::steady_clock::now()+chrono::milliseconds((int)(max(0.01,secs-0.15)*1000));
  st.resize(N);ind.resize(N);ms.resize(N);mp.resize(N);q.resize(N);
  vector<vector<vector<int>>> elites; vector<ll> ev;
  for(int k=0;k<128&&!timeup();k++){auto o=dispatch(k);ll val;decode(o,val,true);elites.push_back(o);ev.push_back(val);}
  vector<int> sorted(elites.size());iota(sorted.begin(),sorted.end(),0);sort(sorted.begin(),sorted.end(),[&](int a,int b){return ev[a]<ev[b];});
  vector<vector<int>> order=best;ll value;double cur=decode(order,value);vector<int> tabu(N*N,0);int iter=0,stall=0,epoch=0;double local=cur;
  vector<char> crit(N);vector<int> pos(N);
  while(!timeup()){
   iter++;iters++;decode(order,value);fill(crit.begin(),crit.end(),0);
   vector<int> stack;for(int j=0;j<n;j++)if(st[last[j]]+p[last[j]]>d[j]){stack.push_back(last[j]);crit[last[j]]=1;}
   if(stack.empty())for(int j=0;j<n;j++){stack.push_back(last[j]);crit[last[j]]=1;}
   for(int h=0;h<(int)stack.size();h++){int v=stack[h];for(int u:{pre[v],mp[v]})if(u>=0&&st[u]+p[u]==st[v]&&!crit[u]){crit[u]=1;stack.push_back(u);}}
   for(auto&a:order)for(int k=0;k<(int)a.size();k++)pos[a[k]]=k;
   double score=1e100;int bm=-1,bk=0,bl=0;ll selected=LLONG_MAX;
   int radius=(stall>40? n:8);
   for(int a=0;a<m;a++){
    auto &row=order[a];int len=row.size();
    for(int k=0;k<len;k++){
     int v=row[k];if(!crit[v]&&iter%7!=0)continue;
     int lo=max(0,k-radius),hi=min(len-1,k+radius);
     for(int l=lo;l<=hi;l++)if(l!=k){
      if(!crit[row[l]]&&!crit[v])continue;
      bool forbidden=false;
      for(int h=min(k,l);h<=max(k,l);h++)if(h!=k){int b=row[h];int x=k<l?b:v,y=k<l?v:b;if(tabu[x*N+y]>iter)forbidden=true;}
      move(row,k,l);ll nv;double ns=decode(order,nv);
      if(nv<bv){bv=nv;best=order;bst=st;bestiter=iters;}
      if(ns<score&&(!forbidden||nv<bv||nv==bv&&ns<cur)){score=ns;bm=a;bk=k;bl=l;selected=nv;}
      move(row,l,k);
     }
     if(timeup())break;
    }if(timeup())break;
   }
   if(bm>=0){auto &row=order[bm];int v=row[bk];int tenure=7+rng()%12;
    for(int h=min(bk,bl);h<=max(bk,bl);h++)if(h!=bk){int b=row[h];int x=bk<bl?v:b,y=bk<bl?b:v;tabu[x*N+y]=iter+tenure;}
    move(row,bk,bl);cur=score;
    if(cur<local-1e-7){local=cur;stall=0;}else stall++;
   }else stall+=30;
   if(stall>100||bm<0){
    epoch++; if(epoch%5==0)order=dispatch(2+epoch);else if(epoch%3==0&&!sorted.empty())order=elites[sorted[rng()%min((int)sorted.size(),12)]];else order=best;
    int changes=3+rng()%18;
    for(int h=0;h<changes;h++){int a=rng()%m,k=rng()%order[a].size(),l=rng()%order[a].size();move(order[a],k,l);ll nv;decode(order,nv);if(nv==LLONG_MAX)move(order[a],l,k);}
    fill(tabu.begin(),tabu.end(),0);cur=decode(order,value,true);local=cur;stall=0;
   }
  }
  cerr<<"objective="<<bv<<" evaluations="<<evals<<" iterations="<<iters<<" best_iteration="<<bestiter<<" seed="<<seed<<" workers=1\n";
 }
};
int main(int argc,char**argv){if(argc<6)return 2;int workers=stoi(argv[5]);if(workers!=1){cerr<<"This implementation requires workers=1\n";return 3;}
 ifstream in(argv[1]);Solver s;in>>s.n>>s.m;s.N=s.n*s.m;s.nodes.resize(s.n);
 for(int j=0;j<s.n;j++){int id,r,d,w;in>>id>>r>>d>>w;s.jid.push_back(id);s.r.push_back(r);s.d.push_back(d);s.w.push_back(w);for(int k=0;k<s.m;k++){int a,p;in>>a>>p;int v=s.p.size();s.nodes[j].push_back(v);s.p.push_back(p);s.mac.push_back(a);s.own.push_back(j);s.pre.push_back(k? v-1:-1);s.suc.push_back(k+1<s.m?v+1:-1);s.base.push_back(k?0:r);}s.last.push_back(s.p.size()-1);}
 s.tail.resize(s.N);for(int j=0;j<s.n;j++)for(int k=s.m-1;k>=0;k--){int v=s.nodes[j][k];s.tail[v]=s.p[v]+(s.suc[v]<0?0:s.tail[s.suc[v]]);}
 s.run(stod(argv[3]),stoi(argv[4]));ofstream out(argv[2]);out<<"{\"objective_value\":"<<s.bv<<",\"schedule\":[";
 for(int j=0;j<s.n;j++){if(j)out<<",";int c=s.bst[s.last[j]]+s.p[s.last[j]];out<<"{\"job_id\":"<<s.jid[j]<<",\"completion_time\":"<<c<<",\"tardiness\":"<<max(0,c-s.d[j])<<",\"operations\":[";for(int k=0;k<s.m;k++){if(k)out<<",";int v=s.nodes[j][k];out<<"{\"machine\":"<<s.mac[v]<<",\"start_time\":"<<s.bst[v]<<"}";}out<<"]}";}out<<"]}\n";
}
