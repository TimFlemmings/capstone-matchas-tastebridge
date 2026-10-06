import type { Link } from "./state/store";

/** The recommender's connecting factors, in words. */
const FACTOR_LABELS: Record<string, string> = {
    genre: "same genre",
    genre_lineage: "genre family",
    related_genre: "related genre",
    era: "same era",
    geography: "same region",
};

export function linkParts(link?: Link): string[] {
    if (!link) return [];
    const parts = link.factors.map((factor) => FACTOR_LABELS[factor] ?? factor);
    if (link.distance !== undefined) parts.push(`${Math.round(link.distance * 100)} bridgons`);
    return parts;
}
