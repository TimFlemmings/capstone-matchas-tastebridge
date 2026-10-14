import { useEffect, useMemo, useRef, useState } from "react";
import { getJourneyMap, type MapPoint, type Track } from "../api/recommendations";
import { linkParts } from "../connections";
import type { JourneyNode, SavedTrack } from "../state/store";

const tierName = (tier: string | null) => tier === "Familiar" ? "Home" : tier === "Adventurous" ? "Adventure" : tier;

// label type, measured in the real fonts so placement can avoid overlaps
const SEED_FONT = "700 12.5px Manrope, sans-serif";
const SAVED_FONT = "500 11px Manrope, sans-serif";
const STEP_FONT = "500 10px 'DM Mono', monospace";
const MIN_GAP = 76;  // closest two dots may sit, in px

type Box = { x: number; y: number; w: number; h: number };
type Label = { x: number; y: number; anchor: "start" | "middle" | "end" };

let measureContext: CanvasRenderingContext2D | null = null;
function textWidth(text: string, font: string) {
    measureContext ??= document.createElement("canvas").getContext("2d");
    if (!measureContext) return text.length * 7;
    measureContext.font = font;
    return measureContext.measureText(text).width;
}

const overlaps = (a: Box, b: Box) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
const overlapArea = (a: Box, boxes: Box[]) => boxes.reduce((sum, b) => sum
    + Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x)) * Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y)), 0);

/** Where a label sits for each candidate side of a dot of radius r. */
function labelCandidates(x: number, y: number, r: number, width: number, height: number): [Label, Box][] {
    const gap = r + 6;
    const side = (dx: number, dy: number, anchor: Label["anchor"]): [Label, Box] => {
        const left = anchor === "start" ? x + dx : anchor === "end" ? x + dx - width : x + dx - width / 2;
        return [{ x: x + dx, y: y + dy + height / 3, anchor }, { x: left, y: y + dy - height / 2, w: width, h: height }];
    };
    return [
        side(gap, 0, "start"), side(-gap, 0, "end"),
        side(0, gap + height * .5, "middle"), side(0, -gap - height * .5, "middle"),
        side(gap, gap + 4, "start"), side(gap, -gap - 4, "start"), side(-gap, gap + 4, "end"), side(-gap, -gap - 4, "end"),
        side(0, gap * 2 + height, "middle"), side(0, -gap * 2 - height, "middle"),
    ];
}

type Pt = { x: number; y: number };

/** Steps are drawn as gentle curves bowing to the left of travel, so two steps
 *  along the same path (A to B, then B back towards A) never sit on top of each other. */
function curveControl(a: Pt, b: Pt): Pt {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const length = Math.hypot(dx, dy) || 1;
    const bend = Math.min(42, length * 0.16);
    return { x: (a.x + b.x) / 2 - (dy / length) * bend, y: (a.y + b.y) / 2 + (dx / length) * bend };
}

function curvePoint(a: Pt, c: Pt, b: Pt, t: number): Pt {
    const u = 1 - t;
    return { x: u * u * a.x + 2 * u * t * c.x + t * t * b.x, y: u * u * a.y + 2 * u * t * c.y + t * t * b.y };
}

function curveAngle(a: Pt, c: Pt, b: Pt, t: number): number {
    const dx = 2 * (1 - t) * (c.x - a.x) + 2 * t * (b.x - c.x);
    const dy = 2 * (1 - t) * (c.y - a.y) + 2 * t * (b.y - c.y);
    let angle = (Math.atan2(dy, dx) * 180) / Math.PI;
    if (angle > 90) angle -= 180;
    if (angle < -90) angle += 180;
    return angle;
}

/** Your seeds and saved tracks placed by how they sound: closer = more alike.
 *  Lines are the steps you took (seed to seed, coloured by tier) and the tracks
 *  you saved (dotted, to the seed they came from). Only seeds and saved tracks
 *  appear. The canvas grows with the journey and scrolls when it outgrows the page. */
export default function JourneyMap({ nodes, saved, currentKey, nowPlayingId, onRevisit, onPlay }: {
    nodes: JourneyNode[];
    saved: SavedTrack[];
    currentKey: string | null;
    nowPlayingId: string;
    onRevisit: (node: JourneyNode) => void;
    onPlay: (track: Track) => void;
}) {
    const keys = useMemo(() => new Set(nodes.map((node) => node.key)), [nodes]);
    const savedHere = useMemo(() => saved.filter((item) => item.seedKey && keys.has(item.seedKey)), [saved, keys]);
    const ids = useMemo(() => [...new Set([...nodes.map((node) => node.track.id), ...savedHere.map((item) => item.track.id)])], [nodes, savedHere]);
    const [points, setPoints] = useState<Map<string, MapPoint> | null>(null);
    const [failed, setFailed] = useState(false);
    const [fontsReady, setFontsReady] = useState(false);

    useEffect(() => {
        document.fonts?.ready.then(() => setFontsReady(true));
    }, []);

    useEffect(() => {
        let active = true;
        getJourneyMap(ids)
            .then((found) => {
                if (!active) return;
                setPoints(new Map(found.map((point) => [point.id, point])));
                setFailed(false);
            })
            .catch(() => active && setFailed(true));
        return () => {
            active = false;
        };
    }, [ids.join(",")]);  // eslint-disable-line react-hooks/exhaustive-deps

    const layout = useMemo(() => {
        if (!points) return null;
        const seedIds = new Set(nodes.map((node) => node.track.id));
        const placed = ids.filter((id) => points.has(id));
        // the canvas grows with the journey so dots and names have room
        const W = Math.max(940, Math.round(260 * Math.sqrt(placed.length) + 300));
        const H = Math.max(560, Math.round(W * 0.62));
        // halfway between the true layout and an even spread: keeps who is left/right
        // and above/below of whom, but stops one outlier squeezing everyone else
        const evenly = (axis: "x" | "y") => {
            const order = [...placed].sort((a, b) => points.get(a)![axis] - points.get(b)![axis]);
            return new Map(order.map((id, rank) => [id, 0.08 + 0.84 * (placed.length > 1 ? rank / (placed.length - 1) : 0.5)]));
        };
        const evenX = evenly("x");
        const evenY = evenly("y");
        const pos = new Map(placed.map((id) => [id, {
            x: (points.get(id)!.x + evenX.get(id)!) / 2 * W,
            y: (points.get(id)!.y + evenY.get(id)!) / 2 * H,
        }]));

        // nudge apart any dots closer than MIN_GAP, keeping the overall shape
        for (let round = 0; round < 80; round += 1) {
            let moved = false;
            for (let i = 0; i < placed.length; i += 1) {
                for (let j = i + 1; j < placed.length; j += 1) {
                    const a = pos.get(placed[i])!;
                    const b = pos.get(placed[j])!;
                    const dx = b.x - a.x;
                    const dy = b.y - a.y;
                    const dist = Math.hypot(dx, dy) || 0.01;
                    if (dist >= MIN_GAP) continue;
                    const push = (MIN_GAP - dist) / 2;
                    const ux = dist > 0.01 ? dx / dist : Math.cos(i + j);
                    const uy = dist > 0.01 ? dy / dist : Math.sin(i + j);
                    a.x -= ux * push; a.y -= uy * push;
                    b.x += ux * push; b.y += uy * push;
                    moved = true;
                }
            }
            for (const p of pos.values()) {
                p.x = Math.min(W - 60, Math.max(60, p.x));
                p.y = Math.min(H - 40, Math.max(40, p.y));
            }
            if (!moved) break;
        }

        // labels: seeds first (they matter most), each on the first side that's free
        const occupied: Box[] = [...pos.entries()].map(([id, p]) => {
            const r = seedIds.has(id) ? 12 : 7;
            return { x: p.x - r, y: p.y - r, w: r * 2, h: r * 2 };
        });
        const labels = new Map<string, Label>();

        // the lines on the map, sampled every few px, so names can keep off them
        const segments: [number, number, number, number][] = [];
        for (const node of nodes) {
            const parent = node.parentKey && nodes.find((item) => item.key === node.parentKey);
            const a = parent && pos.get(parent.track.id);
            const b = pos.get(node.track.id);
            if (!a || !b) continue;
            const c = curveControl(a, b);
            for (let i = 0; i < 12; i += 1) {
                const p0 = curvePoint(a, c, b, i / 12);
                const p1 = curvePoint(a, c, b, (i + 1) / 12);
                segments.push([p0.x, p0.y, p1.x, p1.y]);
            }
        }
        for (const item of savedHere) {
            const seed = nodes.find((node) => node.key === item.seedKey);
            const a = seed && pos.get(seed.track.id);
            const b = pos.get(item.track.id);
            if (a && b) segments.push([a.x, a.y, b.x, b.y]);
        }
        const linePoints = segments.flatMap(([x1, y1, x2, y2]) => {
            const n = Math.max(2, Math.ceil(Math.hypot(x2 - x1, y2 - y1) / 5));
            return Array.from({ length: n + 1 }, (_, i) => [x1 + (x2 - x1) * i / n, y1 + (y2 - y1) * i / n]);
        });
        const lineHits = (b: Box) => linePoints.reduce((hits, [x, y]) => hits + (x > b.x && x < b.x + b.w && y > b.y && y < b.y + b.h ? 1 : 0), 0);

        const placeLabels = (group: string[]) => {
        for (const id of group) {
            const p = pos.get(id)!;
            const isSeed = seedIds.has(id);
            const title = (isSeed ? nodes.find((node) => node.track.id === id)!.track : savedHere.find((item) => item.track.id === id)!.track).title;
            const width = textWidth(title, isSeed ? SEED_FONT : SAVED_FONT);
            const options = labelCandidates(p.x, p.y, isSeed ? 11 : 6, width, isSeed ? 15 : 13);
            const own = occupied.filter((box) => !(box.x <= p.x && p.x <= box.x + box.w && box.y <= p.y && p.y <= box.y + box.h));
            const onCanvas = (b: Box) => b.x >= 6 && b.y >= 6 && b.x + b.w <= W - 6 && b.y + b.h <= H - 6;
            const cost = (b: Box) => overlapArea(b, own) + 60 * lineHits(b) + (onCanvas(b) ? 0 : 1e6);
            const [label, box] = options.find(([, b]) => onCanvas(b) && !own.some((o) => overlaps(b, o)) && lineHits(b) === 0)
                ?? options.reduce((best, option) => cost(option[1]) < cost(best[1]) ? option : best);
            labels.set(id, label);
            occupied.push(box);
        }
        };

        // order matters: seed names first, then distances, then saved-track names steer around both
        placeLabels(placed.filter((id) => seedIds.has(id)));

        // distance labels slide along their line to a free spot
        const steps = new Map<string, { x: number; y: number; angle: number; side: number }>();
        for (const node of nodes) {
            if (!node.parentKey || node.link?.distance === undefined) continue;
            const parent = nodes.find((item) => item.key === node.parentKey);
            const from = parent && pos.get(parent.track.id);
            const to = pos.get(node.track.id);
            if (!from || !to) continue;
            const control = curveControl(from, to);
            const text = `${Math.round(node.link.distance * 100)} bridgons`;
            const w = textWidth(text, STEP_FONT) + 4;
            const h = 13;
            // along the curve, either side, next to it or (if crowded) a little further off
            const options = [8, 20, 32].flatMap((offset) => [0.5, 0.45, 0.55, 0.4, 0.6, 0.35, 0.65, 0.3, 0.7, 0.25, 0.75, 0.2, 0.8].flatMap((t) => [-1, 1].map((side) => {
                const at = curvePoint(from, control, to, t);
                const angle = curveAngle(from, control, to, t);
                const rad = (angle * Math.PI) / 180;
                const bw = Math.abs(w * Math.cos(rad)) + Math.abs(h * Math.sin(rad));
                const bh = Math.abs(w * Math.sin(rad)) + Math.abs(h * Math.cos(rad));
                // same offset the text is drawn with: translate(0, d) after rotating moves it by d·(-sin, cos)
                const cx = at.x - side * offset * Math.sin(rad);
                const cy = at.y + side * offset * Math.cos(rad);
                return { at, angle, side: side * offset, box: { x: cx - bw / 2, y: cy - bh / 2, w: bw, h: bh } };
            })));
            const best = options.find((option) => !occupied.some((o) => overlaps(option.box, o)))
                ?? options.reduce((a, b) => overlapArea(a.box, occupied) <= overlapArea(b.box, occupied) ? a : b);
            occupied.push(best.box);
            steps.set(node.key, { x: best.at.x, y: best.at.y, angle: best.angle, side: best.side });
        }
        placeLabels(placed.filter((id) => !seedIds.has(id)));

        return { W, H, pos, labels, steps };
    }, [points, ids, nodes, savedHere, fontsReady]);  // eslint-disable-line react-hooks/exhaustive-deps

    const scrollRef = useRef<HTMLDivElement>(null);
    const currentId = nodes.find((node) => node.key === currentKey)?.track.id;
    useEffect(() => {
        const box = scrollRef.current;
        const p = layout && currentId ? layout.pos.get(currentId) : undefined;
        if (box && p) box.scrollTo({ left: p.x - box.clientWidth / 2, top: p.y - box.clientHeight / 2 });
    }, [layout, currentId]);

    if (failed) return <div className="error-state">The map couldn't be laid out just now.</div>;
    if (!layout) return <div className="empty-state">Laying out your map…</div>;

    const { W, H, pos, labels, steps } = layout;
    const byKey = new Map(nodes.map((node) => [node.key, node]));
    const seedTrackIds = new Set(nodes.map((node) => node.track.id));
    const labelFor = (id: string, text: string, className: string) => {
        const label = labels.get(id);
        return label && <text x={label.x} y={label.y} textAnchor={label.anchor} className={className}>{text}</text>;
    };

    return (
        <div className="journey-map">
            <div className="journey-map-scroll" ref={scrollRef}>
                <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Your seeds and saved tracks, placed by how they sound">
                    <defs>
                        <linearGradient id="map-accent" x1="0" y1="0" x2="1" y2="1">
                            <stop offset="0" stopColor="var(--accent-2)" />
                            <stop offset="1" stopColor="var(--accent)" />
                        </linearGradient>
                    </defs>

                    {/* saved tracks hang off their seed */}
                    {savedHere.map(({ track, seedKey, link }) => {
                        const from = pos.get(byKey.get(seedKey!)!.track.id);
                        const to = pos.get(track.id);
                        if (!from || !to) return null;
                        return (
                            <line key={`s-${seedKey}-${track.id}`} x1={from.x} y1={from.y} x2={to.x} y2={to.y} className="map-saved-link">
                                <title>{`${track.title} (saved) · ${linkParts(link).join(" · ") || "saved from this seed"}`}</title>
                            </line>
                        );
                    })}

                    {/* the steps between seeds */}
                    {nodes.filter((node) => node.parentKey).map((node) => {
                        const from = pos.get(byKey.get(node.parentKey!)!.track.id);
                        const to = pos.get(node.track.id);
                        if (!from || !to) return null;
                        const label = [tierName(node.tier), ...linkParts(node.link)].filter(Boolean).join(" · ");
                        const spot = steps.get(node.key);
                        return (
                            <g key={`e-${node.key}`} className={`map-step tier-${(node.tier ?? "start").toLowerCase()}`}>
                                <title>{label}</title>
                                <path d={`M${from.x},${from.y} Q${curveControl(from, to).x},${curveControl(from, to).y} ${to.x},${to.y}`} />
                                <path d={`M${from.x},${from.y} Q${curveControl(from, to).x},${curveControl(from, to).y} ${to.x},${to.y}`} className="map-step-hit" />
                                {spot && node.link?.distance !== undefined && (
                                    <text transform={`translate(${spot.x} ${spot.y}) rotate(${spot.angle}) translate(0 ${spot.side < 0 ? spot.side + 2 : spot.side + 6})`} textAnchor="middle">
                                        {Math.round(node.link.distance * 100)} bridgons
                                    </text>
                                )}
                            </g>
                        );
                    })}

                    {savedHere.map(({ track, link }) => {
                        const p = pos.get(track.id);
                        if (!p || seedTrackIds.has(track.id)) return null;  // saved and later seeded: the seed dot stands for it
                        return (
                            <g key={`sd-${track.id}`} className={nowPlayingId === track.id ? "map-saved playing" : "map-saved"}
                                onClick={() => onPlay(track)} role="button" tabIndex={0}
                                onKeyDown={(event) => event.key === "Enter" && onPlay(track)}>
                                <title>{`${track.title} – ${track.artist} · ${linkParts(link).join(" · ")} (click to play)`}</title>
                                <circle cx={p.x} cy={p.y} r="6" />
                                {labelFor(track.id, track.title, "map-label saved")}
                            </g>
                        );
                    })}

                    {nodes.map((node) => {
                        const p = pos.get(node.track.id);
                        // a track seeded more than once gets one dot: the current visit, else the first
                        const shown = nodes.find((item) => item.track.id === node.track.id && item.key === currentKey)
                            ?? nodes.find((item) => item.track.id === node.track.id);
                        if (!p || shown !== node) return null;
                        const current = node.key === currentKey;
                        return (
                            <g key={`n-${node.key}`} className={current ? "map-seed current" : "map-seed"}
                                onClick={() => onRevisit(node)} role="button" tabIndex={0}
                                onKeyDown={(event) => event.key === "Enter" && onRevisit(node)}>
                                <title>{`${node.track.title} – ${node.track.artist} · ${node.track.genre} (click to explore from here)`}</title>
                                <circle cx={p.x} cy={p.y} r={current ? 11 : 9} />
                                {labelFor(node.track.id, node.track.title, "map-label seed")}
                            </g>
                        );
                    })}
                </svg>
            </div>
            <div className="journey-map-key">
                <div className="key-legend">
                    <span><i className="key-seed"></i>Seed</span>
                    <span><i className="key-saved"></i>Saved track</span>
                    <span><i className="key-dotted"></i>Saved from this seed</span>
                    <span><i className="key-line tier-familiar"></i>Home</span>
                    <span><i className="key-line tier-bridge"></i>Bridge</span>
                    <span><i className="key-line tier-adventurous"></i>Adventure</span>
                    <span className="key-note">Closer together = more alike in sound</span>
                </div>
                {steps.size > 0 && (
                    <ol className="key-steps" aria-label="Your steps and why they connect">
                        {nodes.filter((node) => node.parentKey).map((node) => {
                            const parent = byKey.get(node.parentKey!);
                            if (!parent) return null;
                            const tier = (node.tier ?? "start").toLowerCase();
                            return (
                                <li key={node.key}>
                                    <i className={`key-line tier-${tier}`}></i>
                                    <span className="key-step-route"><strong>{parent.track.title}</strong> → <strong>{node.track.title}</strong></span>
                                    <span className="key-step-why">{[tierName(node.tier), ...linkParts(node.link)].filter(Boolean).join(" · ")}</span>
                                </li>
                            );
                        })}
                    </ol>
                )}
            </div>
        </div>
    );
}
