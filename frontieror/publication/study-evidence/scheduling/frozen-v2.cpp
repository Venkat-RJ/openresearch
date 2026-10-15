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
using namespace std; using ll=long long; using Order=vector<vector<int>>;
struct Solver {
 int n,m,N,stamp=0; vector<int> jid,r,d,w,p,mac,pre,suc,last,base,tail,mp,st,seen,done,jseq;
 vector<vector<int>> nodes; mt19937 rng; chrono::steady_clock::time_point deadline;
 ll evals=0,cuts=0,cycles=0,iters=0,bv=LLONG_MAX; vector<int> bst; Order best;
 vector<Order> elites; vector<ll> escores;
 double U(){return (rng()+0.5)/4294967296.0;}
 bool timeup(){return chrono::steady_clock::now()>=deadline;}
 void links(const Order&o){fill(mp.begin(),mp.end(),-1);for(auto&a:o)for(int k=1;k<(int)a.size();k++)mp[a[k]]=a[k-1];}
 void rowlinks(const vector<int>&a){for(int k=0;k<(int)a.size();k++)mp[a[k]]=k?a[k-1]:-1;}
 int visit(int v){
  if(seen[v]==stamp){if(done[v]!=stamp)return -1;return st[v];}
  seen[v]=stamp;int t=base[v];
  int u=pre[v];if(u>=0){int a=visit(u);if(a<0)return -1;t=max(t,a+p[u]);}
  u=mp[v];if(u>=0){int a=visit(u);if(a<0)return -1;t=max(t,a+p[u]);}
  st[v]=t;done[v]=stamp;return t;
 }
 double decode(ll&val,double bound=1e100){
  ++evals;++stamp;if(stamp==INT_MAX){fill(seen.begin(),seen.end(),0);fill(done.begin(),done.end(),0);stamp=1;}
  val=0;ll flow=0;
  for(int j:jseq){int t=visit(last[j]);if(t<0){++cycles;val=LLONG_MAX;return 1e100;}int c=t+p[last[j]];val+=(ll)w[j]*max(0,c-d[j]);flow+=(ll)w[j]*c;if((double)val>bound){++cuts;val=LLONG_MAX;return 1e100;}}
  return val+0.000001*flow;
 }
 void keep(const Order&o,ll val){if(val<bv){bv=val;best=o;bst=st;}}
 Order dispatch(int run){
  Order o(m);vector<int>k(n),ready=r,mr(m);double avg=accumulate(p.begin(),p.end(),0.0)/N;
  double K=run==0?2.0:exp(log(0.3)+U()*log(20.0));double noise=run<2?0:0.1+U()*0.9;
  for(int step=0;step<N;step++){
   int pivot=-1,finish=INT_MAX;vector<int>est(n),v(n,-1);
   for(int j=0;j<n;j++)if(k[j]<(int)nodes[j].size()){v[j]=nodes[j][k[j]];est[j]=max(ready[j],mr[mac[v[j]]]);if(est[j]+p[v[j]]<finish){finish=est[j]+p[v[j]];pivot=j;}}
   int machine=mac[v[pivot]],chosen=pivot;double bs=-1e100;
   for(int j=0;j<n;j++)if(v[j]>=0&&mac[v[j]]==machine&&est[j]<finish){int a=v[j];double sc=log((double)max(w[j],1)/max(1,p[a]))-max(0,d[j]-est[j]-tail[a])/(K*avg);sc+=noise*(-log(-log(U())));if(sc>bs){bs=sc;chosen=j;}}
   int a=v[chosen];o[machine].push_back(a);ready[chosen]=est[chosen]+p[a];mr[machine]=ready[chosen];k[chosen]++;
  }return o;
 }
 void move(vector<int>&a,int k,int l){int v=a[k];if(k<l)for(int h=k;h<l;h++)a[h]=a[h+1];else for(int h=k;h>l;h--)a[h]=a[h-1];a[l]=v;}
 int distance(const Order&a,const Order&b){vector<int>pos(N);for(auto&x:a)for(int k=0;k<(int)x.size();k++)pos[x[k]]=k;int z=0;for(auto&x:b)for(int k=0;k<(int)x.size();k++)z+=abs(k-pos[x[k]]);return z;}
 void addelite(const Order&o,ll v){
  if(v==LLONG_MAX)return;
  for(int h=0;h<(int)elites.size();h++)if(distance(o,elites[h])<N/2){if(v<escores[h]){elites[h]=o;escores[h]=v;}return;}
  if(elites.size()<12){elites.push_back(o);escores.push_back(v);}else {int h=max_element(escores.begin(),escores.end())-escores.begin();if(v<escores[h]){elites[h]=o;escores[h]=v;}}
 }
 void run(double secs,int seed){
  rng.seed(seed);deadline=chrono::steady_clock::now()+chrono::milliseconds((int)(max(.01,secs-.15)*1000));
  st.resize(N);mp.resize(N);seen.assign(N,0);done.assign(N,0);jseq.resize(n);iota(jseq.begin(),jseq.end(),0);
  sort(jseq.begin(),jseq.end(),[&](int a,int b){return w[a]>w[b];});
  for(int k=0;k<128&&!timeup();k++){auto o=dispatch(k);links(o);ll v;decode(v);keep(o,v);addelite(o,v);}
  Order order=best,localorder=order;links(order);ll value;double cur=decode(value),local=cur;ll localval=value;
  vector<int>tabu(N*N),mseq(m);iota(mseq.begin(),mseq.end(),0);vector<char>crit(N);int iter=0,stall=0,epoch=0;
  while(!timeup()){
   ++iter;++iters;decode(value);fill(crit.begin(),crit.end(),0);
   vector<int>stack;for(int j=0;j<n;j++)if(st[last[j]]+p[last[j]]>d[j]){stack.push_back(last[j]);crit[last[j]]=1;}
   if(stack.empty())for(int j=0;j<n;j++){stack.push_back(last[j]);crit[last[j]]=1;}
   for(int h=0;h<(int)stack.size();h++){int v=stack[h];for(int u:{pre[v],mp[v]})if(u>=0&&st[u]+p[u]==st[v]&&!crit[u]){crit[u]=1;stack.push_back(u);}}
   shuffle(mseq.begin(),mseq.end(),rng);double score=1e100;int bm=-1,bk=0,bl=0;int radius=stall>30?n:8;
   for(int a:mseq){auto&row=order[a];int len=row.size();
    for(int k=0;k<len;k++){int v=row[k];if(!crit[v]&&iter%9!=0)continue;
     for(int l=max(0,k-radius);l<=min(len-1,k+radius);l++)if(l!=k){
      if(!crit[row[l]]&&!crit[v])continue;
      bool forbidden=false;for(int h=min(k,l);h<=max(k,l);h++)if(h!=k){int b=row[h],x=k<l?b:v,y=k<l?v:b;if(tabu[x*N+y]>iter){forbidden=true;break;}}
      move(row,k,l);rowlinks(row);ll nv;double bound=forbidden?(double)bv:max((double)bv,score);double ns=decode(nv,bound);
      bool aspir=nv<bv;if(nv!=LLONG_MAX)keep(order,nv);
      if(ns<score&&(!forbidden||aspir)){score=ns;bm=a;bk=k;bl=l;}
      move(row,l,k);rowlinks(row);
     }
     if(timeup())break;
    }if(timeup())break;
   }
   if(bm>=0){auto&row=order[bm];int v=row[bk],tenure=5+rng()%14;
    for(int h=min(bk,bl);h<=max(bk,bl);h++)if(h!=bk){int b=row[h],x=bk<bl?v:b,y=bk<bl?b:v;tabu[x*N+y]=iter+tenure;}
    move(row,bk,bl);rowlinks(row);cur=score;
    if(cur<local-1e-7){local=cur;localorder=order;ll nv;decode(nv);localval=nv;keep(order,nv);stall=0;}else ++stall;
   }else stall+=30;
   if(stall>80||bm<0){
    addelite(localorder,localval);++epoch;
    if(epoch%7==0)order=dispatch(2+epoch);
    else if(epoch%4==0)order=best;
    else {vector<int>z(elites.size());iota(z.begin(),z.end(),0);sort(z.begin(),z.end(),[&](int a,int b){return escores[a]<escores[b];});order=elites[z[rng()%min(8,(int)z.size())]];}
    links(order);int changes=2+rng()%12;
    for(int h=0;h<changes&&!timeup();h++){int a=rng()%m;if(order[a].empty())continue;auto&row=order[a];int k=rng()%row.size(),l;
     if(epoch%3==0)l=rng()%row.size();else l=max(0,min((int)row.size()-1,k+(int)(rng()%9)-4));
     move(row,k,l);rowlinks(row);ll nv;decode(nv);if(nv==LLONG_MAX){move(row,l,k);rowlinks(row);}else keep(order,nv);
    }
    fill(tabu.begin(),tabu.end(),0);cur=decode(value);local=cur;localval=value;localorder=order;stall=0;
   }
  }
  cerr<<"objective="<<bv<<" evals="<<evals<<" cuts="<<cuts<<" cycles="<<cycles<<" iterations="<<iters<<" elites="<<elites.size()<<" seed="<<seed<<" workers=1\n";
 }
};
int main(int argc,char**argv){if(argc<6)return 2;if(stoi(argv[5])!=1){cerr<<"requires workers=1\n";return 3;}
 ifstream in(argv[1]);Solver s;in>>s.n>>s.m;if(!in||s.n<=0||s.m<=0)return 4;s.N=s.n*s.m;s.nodes.resize(s.n);
 for(int j=0;j<s.n;j++){int id,r,d,w;in>>id>>r>>d>>w;s.jid.push_back(id);s.r.push_back(r);s.d.push_back(d);s.w.push_back(w);
  for(int k=0;k<s.m;k++){int a,p;in>>a>>p;int v=s.p.size();s.nodes[j].push_back(v);s.p.push_back(p);s.mac.push_back(a);s.pre.push_back(k?v-1:-1);s.suc.push_back(k+1<s.m?v+1:-1);s.base.push_back(k?0:r);}s.last.push_back(s.p.size()-1);}
 s.tail.resize(s.N);for(int j=0;j<s.n;j++)for(int k=s.m-1;k>=0;k--){int v=s.nodes[j][k];s.tail[v]=s.p[v]+(s.suc[v]<0?0:s.tail[s.suc[v]]);}
 s.run(stod(argv[3]),stoi(argv[4]));ofstream out(argv[2]);out<<"{\"objective_value\":"<<s.bv<<",\"schedule\":[";
 for(int j=0;j<s.n;j++){if(j)out<<",";int c=s.bst[s.last[j]]+s.p[s.last[j]];out<<"{\"job_id\":"<<s.jid[j]<<",\"completion_time\":"<<c<<",\"tardiness\":"<<max(0,c-s.d[j])<<",\"operations\":[";for(int k=0;k<s.m;k++){if(k)out<<",";int v=s.nodes[j][k];out<<"{\"machine\":"<<s.mac[v]<<",\"start_time\":"<<s.bst[v]<<"}";}out<<"]}";}out<<"]}\n";
}
