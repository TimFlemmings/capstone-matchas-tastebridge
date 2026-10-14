import { useId } from "react";

/** The tastey mark: a vinyl record as a planet, with a ring around it. */
export function TastyLogo({ size = 28, tone = "gradient" }: { size?: number; tone?: "gradient" | "white" }) {
    const id = useId();
    const gradient = `${id}-gradient`;
    const front = `${id}-front`;
    // white: for dark backgrounds -- a lighter disc and rim so the record doesn't vanish
    const accent = tone === "white" ? "#ffffff" : `url(#${gradient})`;
    const disc = tone === "white" ? { fill: "#2c2c2c", rim: "#d4d4d4", groove: "#5c5c5c" } : { fill: "#161616", rim: "#3a3a3a", groove: "#444" };
    return (
        <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
            <defs>
                <linearGradient id={gradient} x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0" stopColor="var(--accent-2)" />
                    <stop offset="1" stopColor="var(--accent)" />
                </linearGradient>
                <clipPath id={front}><rect x="0" y="16" width="32" height="16" /></clipPath>
            </defs>
            <g transform="rotate(-22 16 16)">
                <ellipse cx="16" cy="16" rx="15" ry="4.6" fill="none" stroke={accent} strokeWidth="1.6" />
            </g>
            <circle cx="16" cy="16" r="10" fill={disc.fill} stroke={disc.rim} strokeWidth=".9" />
            <circle cx="16" cy="16" r="8.2" fill="none" stroke={disc.groove} strokeWidth=".6" />
            <circle cx="16" cy="16" r="6.6" fill="none" stroke={disc.groove} strokeWidth=".6" />
            <circle cx="16" cy="16" r="3.6" fill={accent} />
            <circle cx="16" cy="16" r=".9" fill={disc.fill} />
            <g transform="rotate(-22 16 16)" clipPath={`url(#${front})`}>
                <ellipse cx="16" cy="16" rx="15" ry="4.6" fill="none" stroke={accent} strokeWidth="1.6" />
            </g>
        </svg>
    );
}

/** Seed / explore-from-here: two notes with sparkles. Drawn like a lucide icon. */
export function NoteStars({ size = 16 }: { size?: number }) {
    return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
            strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M11 18.5V8.5l10-2v10" />
            <circle cx="8" cy="18.5" r="3" />
            <circle cx="18" cy="16.5" r="3" />
            <path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z" transform="translate(0.3 0.3) scale(0.36)" fill="currentColor" stroke="none" />
            <path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z" transform="translate(13.6 -0.4) scale(0.22)" fill="currentColor" stroke="none" />
        </svg>
    );
}

/** Play triangle shifted so its visual centre (centroid) sits in the middle of
 *  the box -- lucide's sits 1.33 units left, which looks off inside a circle. */
export function PlayIcon({ size = 16 }: { size?: number }) {
    return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" strokeWidth="2"
            strokeLinejoin="round" aria-hidden="true">
            <polygon points="7.33 3 21.33 12 7.33 21" />
        </svg>
    );
}
