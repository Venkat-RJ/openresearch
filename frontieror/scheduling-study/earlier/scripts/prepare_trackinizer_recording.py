"""Prepare a supplied Trackinizer screen recording; never read research results."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import subprocess


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inspect(ffmpeg, path):
    probe = subprocess.run([ffmpeg, '-hide_banner', '-i', str(path)],
                           capture_output=True, text=True)
    duration = re.search(r'Duration: (\d+):(\d+):([\d.]+)', probe.stderr)
    dimensions = re.search(r'Video:.*? (\d{3,5})x(\d{3,5})', probe.stderr)
    assert duration and dimensions, probe.stderr
    hours, minutes, seconds = map(float, duration.groups())
    return dict(duration_s=3600 * hours + 60 * minutes + seconds,
                width=int(dimensions[1]), height=int(dimensions[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--ffmpeg', default=shutil.which('ffmpeg'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'dist/assets')
    args = parser.parse_args()
    assert args.ffmpeg and args.source.is_file()
    assets = args.output_dir
    assets.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(args.source)
    source_info = inspect(args.ffmpeg, args.source)
    destination = assets / 'trackinizer-research-replay.mp4'
    command = [args.ffmpeg, '-hide_banner', '-loglevel', 'error',
               '-threads', '1', '-filter_threads', '1', '-hwaccel', 'videotoolbox',
               '-i', str(args.source), '-map', '0:v:0', '-an',
               '-vf', 'scale=2560:-2:flags=lanczos,fps=30',
               '-c:v', 'h264_videotoolbox', '-b:v', '5000k',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               '-map_metadata', '-1', '-y', str(destination)]
    subprocess.run(command, check=True)
    assert sha256(args.source) == source_hash
    web_info = inspect(args.ffmpeg, destination)
    assert abs(web_info['duration_s'] - source_info['duration_s']) < .1
    assert destination.stat().st_size < 25 * 1024 * 1024
    poster = assets / 'trackinizer-research-replay-poster.jpg'
    subprocess.run([args.ffmpeg, '-hide_banner', '-loglevel', 'error', '-threads', '1',
                    '-filter_threads', '1', '-ss', '10', '-i', str(destination),
                    '-frames:v', '1', '-vf', 'scale=1536:-2', '-q:v', '2',
                    '-y', str(poster)], check=True)
    captions = assets / 'trackinizer-research-replay.vtt'
    captions.write_text('''WEBVTT

00:00:00.000 --> 00:00:06.000
Trackinizer Replay at 2x shows record creation order.
The animation does not represent elapsed experiment time.

00:00:06.000 --> 00:00:12.500
Questions, experiments, beliefs and code changes form a shared record.
The graph includes plans and active work, as well as completed studies.

00:00:12.500 --> 00:00:21.510
The opened objective gives Sagent freedom to choose algorithms and experiments.
Recording the work and checking schedules remain mandatory in this campaign.
''')
    manifest = dict(
        kind='user-supplied screen recording of the actual Trackinizer UI',
        recorded_by='Venkat', source_filename=args.source.name,
        source_sha256=source_hash, source_bytes=args.source.stat().st_size,
        source_video=source_info, web_video=web_info,
        preparation=dict(trimmed=False, retimed=False, fabricated_frames=False,
                         width=2560, frames_per_second=30, audio='omitted in the web copy',
                         encoder='h264_videotoolbox', bitrate='5000k'),
        observed_ui=dict(replay_speed='2x', nodes=173, edges=284,
                         opened_issue='95155afc-8a60-4aa3-9f27-62922fe1ba9b'),
        interpretation='Stored record creation order using relations present at recording time, and campaign planning. Animation spacing is a UI convention; not historical edge chronology, optimization convergence, elapsed experiment time, an optimality proof or a count of completed experiments.',
        captions='Editorial explanation based on inspected source frames, not a speech transcript.',
        final_evaluation_performance_accessed=False,
        assets={path.name:dict(sha256=sha256(path), bytes=path.stat().st_size)
                for path in [destination, poster, captions]})
    (assets / 'trackinizer-research-replay.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(video=str(destination), bytes=destination.stat().st_size,
                         duration_s=web_info['duration_s'], source_preserved=True)))


if __name__ == '__main__':
    main()
