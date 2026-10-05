import { useEffect, useMemo, useRef } from "react";
import { BookmarkCheck, Pause } from "lucide-react";
import type { Track } from "../api/recommendations";
import type { JourneyNode, SavedTrack } from "../state/store";
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

    return (
        <ol className="journey-tree">
            {(childrenOf.get(null) ?? []).map((root) => (
                <Branch key={root.key} node={root} childrenOf={childrenOf} savedBySeed={savedBySeed} {...actions} />
            ))}
        </ol>
    );
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
                    {savedHere.map(({ track }) => (
                        <li key={track.id}>
                            <img src={track.coverUrl} alt="" />
                            <span><strong>{track.title}</strong><small>{track.artist} · {track.genre}</small></span>
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
