import { useEffect, useState, type KeyboardEvent, type MouseEvent } from "react";

const BARS = 64;
const peaksCache = new Map<string, Promise<number[]>>();

/** Loudness of the clip in BARS slices (0.15..1), decoded in the browser.
 *  Cached per URL, so a track is only decoded once per visit. */
function loadPeaks(url: string): Promise<number[]> {
    let peaks = peaksCache.get(url);
    if (!peaks) {
        peaks = (async () => {
            const response = await fetch(url);
            if (!response.ok) throw new Error(`audio ${response.status}`);
            const audio = await new OfflineAudioContext(1, 1, 22050).decodeAudioData(await response.arrayBuffer());
            const samples = audio.getChannelData(0);
            const slice = Math.floor(samples.length / BARS);
            const rms = Array.from({ length: BARS }, (_, bar) => {
                let sum = 0;
                let count = 0;
                for (let i = bar * slice; i < (bar + 1) * slice; i += 8) {
                    sum += samples[i] * samples[i];
                    count += 1;
                }
                return Math.sqrt(sum / Math.max(count, 1));
            });
            const loudest = Math.max(...rms) || 1;
            return rms.map((value) => 0.15 + 0.85 * (value / loudest));
        })();
        peaks.catch(() => peaksCache.delete(url));  // let a later visit retry
        peaksCache.set(url, peaks);
    }
    return peaks;
}

/** The track's waveform as a seek bar: bars before `progress` are lit. */
export default function Waveform({ url, progress, onSeek, className = "" }: {
    url?: string;
    progress: number;  // 0..1
    onSeek: (fraction: number) => void;
    className?: string;
}) {
    const [peaks, setPeaks] = useState<number[] | null>(null);

    useEffect(() => {
        let active = true;
        setPeaks(null);
        if (url) loadPeaks(url).then((values) => active && setPeaks(values)).catch(() => undefined);
        return () => {
            active = false;
        };
    }, [url]);

    const values = peaks ?? Array.from({ length: BARS }, () => 0.18);  // flat line until decoded
    const lit = Math.round(progress * BARS);

    function seekTo(event: MouseEvent<HTMLDivElement>) {
        const box = event.currentTarget.getBoundingClientRect();
        onSeek(Math.min(1, Math.max(0, (event.clientX - box.left) / box.width)));
    }

    function seekByKey(event: KeyboardEvent<HTMLDivElement>) {
        if (event.key === "ArrowRight") onSeek(Math.min(1, progress + 0.05));
        else if (event.key === "ArrowLeft") onSeek(Math.max(0, progress - 0.05));
        else return;
        event.preventDefault();
    }

    return (
        <div className={`waveform ${peaks ? "" : "loading"} ${className}`} onClick={seekTo} onKeyDown={seekByKey}
            role="slider" tabIndex={0} aria-label="Playback position" aria-valuemin={0} aria-valuemax={100}
            aria-valuenow={Math.round(progress * 100)}>
            {values.map((value, index) => (
                <span key={index} className={index < lit ? "played" : undefined}
                    style={{ height: `${value * 100}%`, backgroundPosition: `${(index / (BARS - 1)) * 100}% 0` }} />
            ))}
        </div>
    );
}
