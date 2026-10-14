import { Fragment, useEffect, useMemo, useRef } from "react";
import { BookmarkCheck, Pause } from "lucide-react";
import type { Track } from "../api/recommendations";
import type { JourneyNode, Origin, SavedTrack } from "../state/store";
import { linkParts } from "../connections";
import { NoteStars, PlayIcon } from "./icons";

type Actions = {
    currentKey: string | null;
    nowPlayingId: string;
    playing: boolean;
    onRevisit: (node: JourneyNode) => void;
    onPlay: (track: Track) => void;
    onUnsave: (track: Track) => void;
};

/** Your history as a tree, read top to bottom: each seed, the tracks you saved
 *  while it was the seed, then the seeds you went on to from it. */
export default function TasteTree({ nodes, saved, ...actions }: Actions & { nodes: JourneyNode[]; saved: SavedTrack[] }) {
    const childrenOf = useMemo(() => {
        const map = new Map<string | null, JourneyNode[]>();
        for (const node of nodes) map.set(node.parentKey, [...(map.get(node.parentKey) ?? []), node]);
        return map;
    }, [nodes]);
    const savedBySeed = useMemo(() => {
        const map = new Map<string, SavedTrack[]>();
        for (const item of saved) {
            if (item.seedKey) map.set(item.seedKey, [...(map.get(item.seedKey) ?? []), item]);
        }
        return map;
    }, [saved]);

    const roots = childrenOf.get(null) ?? [];
    return (
        <ol className="journey-tree">
            {roots.map((root, index) => (
                <Fragment key={root.key}>
                    {/* several journeys: say how each one began, so a later tree doesn't look like a stray */}
                    {roots.length > 1 && (
                        <li className="journey-heading">
                            <strong>Journey {index + 1}</strong> · {describeOrigin(root.origin, index)}
                        </li>
                    )}
                    <Branch node={root} childrenOf={childrenOf} savedBySeed={savedBySeed} {...actions} />
                </Fragment>
            ))}
        </ol>
    );
}

function describeOrigin(origin: Origin | undefined, index: number): string {
    switch (origin?.via) {
        case "default": return "started with the app's first track";
        case "search": return origin.query ? `jumped here from a search for “${origin.query}”` : "jumped here from a search";
        case "genre": return `picked from ${origin.genre}`;
        case "starter": return "picked from the starting points";
        case "surprise": return "a surprise pick";
        // journeys from before the app recorded how they began
        default: return index === 0 ? "where you started" : "jumped here from a search or a pick";
    }
}

function Branch({ node, childrenOf, savedBySeed, ...actions }: Actions & {
    node: JourneyNode;
    childrenOf: Map<string | null, JourneyNode[]>;
    savedBySeed: Map<string, SavedTrack[]>;
}) {
    const { currentKey, nowPlayingId, playing, onRevisit, onPlay, onUnsave } = actions;
    const isCurrent = node.key === currentKey;
    const ref = useRef<HTMLDivElement>(null);
    useEffect(() => {
        if (isCurrent) ref.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }, [isCurrent]);
    const kids = childrenOf.get(node.key) ?? [];
    const savedHere = savedBySeed.get(node.key) ?? [];
    const tier = (node.tier ?? "start").toLowerCase();

    const playButton = (track: Track, small = false) => {
        const isPlaying = playing && nowPlayingId === track.id;
        return (
            <button className={small ? "journey-play small" : "journey-play"} onClick={() => onPlay(track)} disabled={!track.audioUrl}
                aria-label={`${isPlaying ? "Pause" : "Play"} ${track.title}`}>
                {isPlaying ? <Pause size={small ? 10 : 12} fill="currentColor" /> : <PlayIcon size={small ? 10 : 12} />}
            </button>
        );
    };

    return (
        <li className={`journey-branch tier-${tier}`}>
            <div ref={ref} className={isCurrent ? "journey-seed current" : "journey-seed"}>
                <button className="journey-seed-main" onClick={() => onRevisit(node)} title="Go back to this seed">
                    <img src={node.track.coverUrl} alt="" />
                    <span className="journey-seed-copy">
                        <span className="journey-seed-labels">
                            <span className={`tier-chip tier-${tier}`}>{node.tier === "Familiar" ? "Home" : node.tier === "Adventurous" ? "Adventure" : node.tier ?? "Start"}</span>
                            {linkParts(node.link).length > 0 && <span className="link-text">{linkParts(node.link).join(" · ")}</span>}
                            {isCurrent && <span className="here-chip"><NoteStars size={10} /> Current seed</span>}
                        </span>
                        <strong>{node.track.title}</strong>
                        <span>{node.track.artist} · {node.track.genre}</span>
                    </span>
                </button>
                {playButton(node.track)}
            </div>

            {savedHere.length > 0 && (
                <ul className="journey-saved" aria-label={`Saved while exploring ${node.track.title}`}>
                    {savedHere.map(({ track, link }) => (
                        <li key={track.id}>
                            <img src={track.coverUrl} alt="" />
                            <span>
                                <strong>{track.title}</strong><small>{track.artist} · {track.genre}</small>
                                {linkParts(link).length > 0 && <span className="link-chips">{linkParts(link).map((part) => <i key={part}>{part}</i>)}</span>}
                            </span>
                            {playButton(track, true)}
                            <button className="icon-button is-saved" onClick={() => onUnsave(track)}
                                aria-label={`Remove ${track.title} from your Library`} title="Remove from Library">
                                <BookmarkCheck size={15} />
                            </button>
                        </li>
                    ))}
                </ul>
            )}

            {kids.length > 0 && (
                <ol className="journey-children">
                    {kids.map((kid) => (
                        <Branch key={kid.key} node={kid} childrenOf={childrenOf} savedBySeed={savedBySeed} {...actions} />
                    ))}
                </ol>
            )}
        </li>
    );
}
