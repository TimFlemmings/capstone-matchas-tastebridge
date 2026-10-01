import { useEffect, useRef, type MouseEvent, type PointerEvent } from "react";
import type { TasteMapPoint } from "../api/recommendations";

type TasteMapCanvasProps = {
    points: TasteMapPoint[];
    listeningPath: string[];
    currentTrackId: string;
    seedTrackId: string;
    recommendationIds: string[];
    selectedId: string | null;
    onSelect: (point: TasteMapPoint) => void;
};

const genreColors: Record<string, string> = {
    "Hip-Hop": "#f47b62",
    Electronic: "#b9f35a",
    Rock: "#73c9b5",
    Pop: "#eabf62",
    Folk: "#9aa5ef",
    Experimental: "#e18acb",
    Instrumental: "#72aee8",
    International: "#d6a17d",
    "Soul-RnB": "#df7d88",
    Jazz: "#8bc6cf",
};

export default function TasteMapCanvas({
    points,
    listeningPath,
    currentTrackId,
    seedTrackId,
    recommendationIds,
    selectedId,
    onSelect,
}: TasteMapCanvasProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const wrapperRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        const wrapper = wrapperRef.current;
        const context = canvas?.getContext("2d");
        if (!canvas || !wrapper || !context || points.length === 0) return;

        const render = () => {
            const bounds = wrapper.getBoundingClientRect();
            const ratio = window.devicePixelRatio || 1;
            canvas.width = Math.floor(bounds.width * ratio);
            canvas.height = Math.floor(bounds.height * ratio);
            canvas.style.width = `${bounds.width}px`;
            canvas.style.height = `${bounds.height}px`;
            context.setTransform(ratio, 0, 0, ratio, 0, 0);
            context.clearRect(0, 0, bounds.width, bounds.height);

            const padding = 28;
            const plotWidth = bounds.width - padding * 2;
            const plotHeight = bounds.height - padding * 2;
            context.strokeStyle = "rgba(180, 194, 176, 0.08)";
            context.lineWidth = 1;
            for (let step = 1; step < 5; step += 1) {
                const x = padding + (plotWidth * step) / 5;
                const y = padding + (plotHeight * step) / 5;
                context.beginPath();
                context.moveTo(x, padding);
                context.lineTo(x, bounds.height - padding);
                context.stroke();
                context.beginPath();
                context.moveTo(padding, y);
                context.lineTo(bounds.width - padding, y);
                context.stroke();
            }

            const positions = new Map<string, { x: number; y: number }>();
            const pointPosition = (point: TasteMapPoint) => ({
                x: padding + point.x * plotWidth,
                y: bounds.height - padding - point.y * plotHeight,
            });
            for (const point of points) positions.set(point.id, pointPosition(point));

            const pathPositions = listeningPath
                .map((id) => positions.get(id))
                .filter((position): position is { x: number; y: number } => Boolean(position));
            if (pathPositions.length > 1) {
                context.beginPath();
                pathPositions.forEach((position, index) => {
                    if (index === 0) context.moveTo(position.x, position.y);
                    else context.lineTo(position.x, position.y);
                });
                context.strokeStyle = "#f47b62";
                context.lineWidth = 2.5;
                context.globalAlpha = 0.85;
                context.stroke();
                context.globalAlpha = 1;
            }

            const recommendations = new Set(recommendationIds);
            for (const point of points) {
                const position = positions.get(point.id)!;
                const isCurrent = point.id === currentTrackId;
                const isSeed = point.id === seedTrackId;
                const isSelected = point.id === selectedId;
                const isRecommended = recommendations.has(point.id);
                const radius = isCurrent || isSelected ? 6 : isSeed ? 5 : isRecommended ? 3.5 : 2.3;

                if (isCurrent || isSeed || isSelected) {
                    context.beginPath();
                    context.arc(position.x, position.y, radius + 4, 0, Math.PI * 2);
                    context.strokeStyle = isCurrent ? "#f3f5ef" : isSelected ? "#73c9b5" : "#b9f35a";
                    context.lineWidth = 1.5;
                    context.stroke();
                }

                context.beginPath();
                context.arc(position.x, position.y, radius, 0, Math.PI * 2);
                context.fillStyle = genreColors[point.genre] ?? "#a1aaa1";
                context.globalAlpha = isRecommended || isCurrent || isSeed || isSelected ? 1 : 0.58;
                context.fill();
                context.globalAlpha = 1;
            }

            for (const [step, id] of listeningPath.entries()) {
                const position = positions.get(id);
                if (!position) continue;
                context.beginPath();
                context.arc(position.x, position.y, 3, 0, Math.PI * 2);
                context.fillStyle = "#f47b62";
                context.fill();
                context.fillStyle = "#f3f5ef";
                context.font = "10px Manrope, sans-serif";
                context.fillText(String(step + 1), position.x + 7, position.y - 6);
            }
        };

        const observer = new ResizeObserver(render);
        observer.observe(wrapper);
        render();
        return () => observer.disconnect();
    }, [points, listeningPath, currentTrackId, seedTrackId, recommendationIds, selectedId]);

    function findPoint(event: PointerEvent<HTMLDivElement> | MouseEvent<HTMLDivElement>) {
        const canvas = canvasRef.current;
        if (!canvas || points.length === 0) return null;
        const bounds = canvas.getBoundingClientRect();
        const padding = 28;
        const plotWidth = bounds.width - padding * 2;
        const plotHeight = bounds.height - padding * 2;
        const localX = event.clientX - bounds.left;
        const localY = event.clientY - bounds.top;
        let closest: TasteMapPoint | null = null;
        let closestDistance = 15 ** 2;
        for (const point of points) {
            const x = padding + point.x * plotWidth;
            const y = bounds.height - padding - point.y * plotHeight;
            const distance = (x - localX) ** 2 + (y - localY) ** 2;
            if (distance < closestDistance) {
                closest = point;
                closestDistance = distance;
            }
        }
        return closest;
    }

    return (
        <div
            className="taste-map-canvas"
            ref={wrapperRef}
            onPointerMove={(event) => {
                event.currentTarget.style.cursor = findPoint(event) ? "pointer" : "crosshair";
            }}
            onPointerLeave={(event) => {
                event.currentTarget.style.cursor = "crosshair";
            }}
            onClick={(event) => {
                const point = findPoint(event);
                if (point) onSelect(point);
            }}
            onKeyDown={(event) => {
                if (!points.length || !["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp", "Enter"].includes(event.key)) return;
                event.preventDefault();
                if (event.key === "Enter" && selectedId) {
                    const point = points.find((candidate) => candidate.id === selectedId);
                    if (point) onSelect(point);
                    return;
                }
                const currentIndex = Math.max(0, points.findIndex((point) => point.id === selectedId));
                const forward = event.key === "ArrowRight" || event.key === "ArrowDown";
                const nextIndex = (currentIndex + (forward ? 1 : points.length - 1)) % points.length;
                onSelect(points[nextIndex]);
            }}
            role="application"
            aria-label="Taste Map. Use arrow keys to move between tracks and Enter to select."
            tabIndex={0}
        >
            <canvas ref={canvasRef} />
        </div>
    );
}
