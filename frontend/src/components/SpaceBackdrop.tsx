/** Faint "solar system" behind each page: orbits with planets sitting on them,
 *  plus hearts and stars, after Tim's sketch. Each page gets its own arrangement.
 *  Coordinates are in a 1000 x 700 viewBox; x < 50 is the strip left of the playlist. */

type Point = [number, number, number];  // x, y, size

type Layout = {
    center: [number, number];  // where the orbits sit
    tilt: number;              // degrees
    planets: number[][];       // per orbit (outermost first): angles in degrees
    hearts: Point[];
    sparkles: Point[];
    dots: [number, number][];  // tiny dot stars
};

const LAYOUTS: Record<string, Layout> = {
    discover: {
        center: [1010, 760], tilt: -38,
        planets: [[-128, -96], [-140, -108, -84], [-118]],
        hearts: [[600, 110, 1.1], [860, 560, 1.5], [26, 640, .9], [24, 250, .7], [930, 96, .8], [700, 640, .7]],
        sparkles: [[480, 60, 1], [700, 150, .7], [960, 470, .6], [790, 650, .9], [30, 150, .8], [18, 420, .5],
            [36, 540, 1.1], [380, 30, .45], [880, 300, .35], [600, 690, .5], [990, 230, .7]],
        dots: [[20, 80], [42, 330], [14, 600], [540, 120], [830, 40], [760, 520], [960, 610], [660, 30], [905, 380]],
    },
    saved: {
        center: [900, 780], tilt: 28,
        planets: [[205, 236], [198, 226, 252], [214]],  // the arcs' visible, upper-left side
        hearts: [[160, 140, 1.4], [420, 560, 1], [760, 90, .9], [26, 470, .8], [600, 650, .7], [960, 300, 1]],
        sparkles: [[90, 470, .8], [560, 210, 1], [940, 330, .6], [300, 70, .7], [24, 120, 1], [36, 330, .5],
            [20, 610, .9], [700, 40, .45], [500, 660, .6], [860, 160, .35]],
        dots: [[16, 220], [44, 560], [230, 640], [640, 120], [880, 60], [360, 300], [780, 400], [980, 520]],
    },
    tree: {
        center: [880, -60], tilt: 20,
        planets: [[110, 135], [100, 122, 150], [128]],
        hearts: [[120, 520, 1.3], [520, 640, .9], [930, 470, 1.1], [24, 300, .8], [700, 300, .7], [330, 420, .6]],
        sparkles: [[260, 150, .9], [640, 300, .7], [80, 360, .6], [800, 620, 1], [30, 80, .7], [20, 620, 1],
            [460, 520, .4], [980, 640, .5], [600, 120, .35], [150, 660, .55]],
        dots: [[40, 440], [14, 180], [380, 600], [720, 520], [900, 280], [560, 400], [240, 300], [990, 120]],
    },
};

const ORBITS: [number, number][] = [[470, 270], [360, 205], [250, 140]];
const PLANET_SIZES = [7, 5.5, 4.5];
const HEART = "M0 -3C-2 -7 -9 -6 -9 -1C-9 4 -2 7 0 10C2 7 9 4 9 -1C9 -6 2 -7 0 -3Z";
const SPARKLE = "M0 -9L2 -2L9 0L2 2L0 9L-2 2L-9 0L-2 -2Z";

export default function SpaceBackdrop({ page }: { page: string }) {
    const layout = LAYOUTS[page] ?? LAYOUTS.discover;
    const [cx, cy] = layout.center;
    return (
        <svg className="space-backdrop" viewBox="0 0 1000 700" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
            <g transform={`translate(${cx} ${cy}) rotate(${layout.tilt})`}>
                {ORBITS.map(([rx, ry], orbit) => (
                    <g key={rx}>
                        <ellipse rx={rx} ry={ry} className="orbit" />
                        {(layout.planets[orbit] ?? []).map((angle, index) => {
                            // a point ON the ellipse (not on a circle of radius rx)
                            const t = (angle * Math.PI) / 180;
                            const size = PLANET_SIZES[orbit] * (index % 2 ? .7 : 1);
                            return <circle key={angle} cx={rx * Math.cos(t)} cy={ry * Math.sin(t)} r={size}
                                className={index % 2 ? "planet small" : "planet"} />;
                        })}
                    </g>
                ))}
            </g>
            {layout.dots.map(([x, y]) => <circle key={`d${x}-${y}`} cx={x} cy={y} r="1.3" className="space-dot" />)}
            {layout.hearts.map(([x, y, size]) => (
                <path key={`h${x}-${y}`} d={HEART} className="space-heart" transform={`translate(${x} ${y}) scale(${size}) rotate(-12)`} />
            ))}
            {layout.sparkles.map(([x, y, size]) => (
                <path key={`s${x}-${y}`} d={SPARKLE} className="space-sparkle" transform={`translate(${x} ${y}) scale(${size})`} />
            ))}
        </svg>
    );
}
