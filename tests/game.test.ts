import { describe, expect, it } from 'vitest';
import { CAR, OBSTACLES, ROAD, SPEED } from '../src/constants';
import { Car } from '../src/game/car';
import { Game } from '../src/game/game';
import { GameLoop } from '../src/game/loop';
import type { FrameScheduler } from '../src/game/loop';
import { ObstacleField, createRng } from '../src/game/obstacles';
import { Road } from '../src/game/road';
import { ScoreKeeper } from '../src/game/score';
import type { Gesture } from '../src/types';

const DT = 1 / 60;

describe('Car', () => {
  it('steers one lane at a time and refuses to leave the road', () => {
    const car = new Car();
    expect(car.lane).toBe(1);
    expect(car.steer(-1)).toBe(true);
    expect(car.steer(-1)).toBe(false);
    expect(car.lane).toBe(0);
    expect(car.steer(1)).toBe(true);
    expect(car.steer(1)).toBe(true);
    expect(car.steer(1)).toBe(false);
    expect(car.lane).toBe(ROAD.laneCount - 1);
  });

  it('eases x to the target lane without overshoot', () => {
    const car = new Car();
    car.steer(1);
    const start = car.x;
    let previous = start;
    for (let i = 0; i < 60; i += 1) {
      car.update(DT);
      expect(car.x).toBeGreaterThanOrEqual(previous);
      previous = car.x;
    }
    expect(previous).toBeGreaterThan(start);
    expect(car.x).toBeCloseTo(start + 120, 0);
  });

  it('jumps once, is airborne for part of the arc, then lands', () => {
    const car = new Car();
    expect(car.jump()).toBe(true);
    expect(car.jump()).toBe(false);
    let airborneFrames = 0;
    let frames = 0;
    while (car.height01 > 0 || frames === 0) {
      car.update(DT);
      frames += 1;
      if (car.airborne) airborneFrames += 1;
      if (frames > 200) throw new Error('jump never ended');
    }
    const airborneS = airborneFrames * DT;
    expect(airborneS).toBeGreaterThan(0.45);
    expect(airborneS).toBeLessThan(0.7);
    expect(car.jump()).toBe(true);
  });
});

describe('Road', () => {
  it('accelerates to, and never beyond, max speed', () => {
    const road = new Road();
    for (let i = 0; i < 60 * 120; i += 1) road.update(DT);
    expect(road.speed).toBe(SPEED.max);
    expect(road.difficulty).toBe(1);
  });
});

describe('ScoreKeeper', () => {
  it('adds metres and pass bonuses', () => {
    const score = new ScoreKeeper();
    score.addDistance(2500);
    score.addPassed(2);
    expect(score.value).toBe(100 + 20);
  });
});

describe('ObstacleField fairness', () => {
  it('never occupies all lanes and keeps spawning rows', () => {
    for (let seed = 1; seed <= 25; seed += 1) {
      const field = new ObstacleField(createRng(seed));
      const road = new Road();
      let rows = 0;
      let lastId = 0;
      for (let i = 0; i < 60 * 150; i += 1) {
        road.update(DT);
        field.update(DT, road);
        const fresh = field.items.filter((o) => o.id > lastId);
        if (fresh.length === 0) continue;
        lastId = Math.max(...fresh.map((o) => o.id));
        rows += 1;
        expect(new Set(fresh.map((o) => o.lane)).size).toBe(fresh.length);
        expect(fresh.length).toBeLessThan(ROAD.laneCount);
      }
      expect(rows).toBeGreaterThan(30);
    }
  });

  it('is deterministic for a given seed', () => {
    const run = (): string => {
      const field = new ObstacleField(createRng(7));
      const road = new Road();
      for (let i = 0; i < 600; i += 1) {
        road.update(DT);
        field.update(DT, road);
      }
      return JSON.stringify(field.items);
    };
    expect(run()).toBe(run());
  });

  it('keeps a free lane within one step of wherever the player can stand (reachability)', () => {
    for (let seed = 1; seed <= 40; seed += 1) {
      const field = new ObstacleField(createRng(seed));
      const road = new Road();
      road.speed = SPEED.max; // double rows are most likely at full difficulty
      let previousFree: number[] = [1];
      const seenRows = new Set<string>();
      let lastId = 0;
      for (let i = 0; i < 60 * 240; i += 1) {
        road.update(DT);
        field.update(DT, road);
        // Select by id: culling removes old obstacles, so array positions shift.
        const fresh = field.items.filter((o) => o.id > lastId);
        if (fresh.length === 0) continue;
        lastId = Math.max(...fresh.map((o) => o.id));
        const carLanes = new Set(fresh.filter((o) => o.kind === 'car').map((o) => o.lane));
        const free = [0, 1, 2].filter((lane) => !carLanes.has(lane));
        for (const from of previousFree) {
          expect(free.some((to) => Math.abs(from - to) <= 1)).toBe(true);
        }
        seenRows.add(fresh.map((o) => `${o.kind}${o.lane}`).join('|'));
        previousFree = free;
      }
      expect(seenRows.size).toBeGreaterThan(3);
    }
  });

  it('lets an airborne car clear a barrier that hits a grounded car', () => {
    const makeRun = (): { car: Car; field: ObstacleField; road: Road } => ({
      car: new Car(),
      field: new ObstacleField(() => 0.1), // 0.1 < barrierChance => barriers only
      road: new Road(),
    });

    const grounded = makeRun();
    let hitStep = -1;
    for (let i = 0; i < 60 * 10 && hitStep < 0; i += 1) {
      grounded.road.update(DT);
      grounded.field.update(DT, grounded.road);
      if (grounded.field.firstHit(grounded.car) !== null) hitStep = i;
    }
    expect(hitStep).toBeGreaterThan(0);
    expect(grounded.field.items.some((o) => o.kind === 'barrier')).toBe(true);

    const jumper = makeRun();
    for (let i = 0; i < hitStep + 60; i += 1) {
      if (i === hitStep - 15) jumper.car.jump();
      jumper.road.update(DT);
      jumper.car.update(DT);
      jumper.field.update(DT, jumper.road);
      expect(jumper.field.firstHit(jumper.car)).toBeNull();
    }
  });
});

describe('Game', () => {
  it('ignores input outside a run', () => {
    const game = new Game({ rng: createRng(1) });
    game.apply('LEFT');
    expect(game.car.lane).toBe(1);
    game.start();
    game.apply('LEFT');
    expect(game.car.lane).toBe(0);
  });

  it('crashes into an obstacle, reports the result once, and freezes the world', () => {
    let overs = 0;
    const game = new Game({ rng: createRng(3), onGameOver: () => { overs += 1; } });
    game.start();
    for (let i = 0; i < 60 * 60 && game.phase === 'playing'; i += 1) game.update(DT);
    expect(game.phase).toBe('gameover');
    expect(overs).toBe(1);
    const frozen = game.score.value;
    for (let i = 0; i < 60; i += 1) game.update(DT);
    expect(game.score.value).toBe(frozen);
    expect(game.lastResult?.score).toBe(frozen);
    expect(game.crashIntensity).toBe(0);
  });

  it('start() fully resets a finished run', () => {
    const game = new Game({ rng: createRng(3) });
    game.start();
    for (let i = 0; i < 60 * 60 && game.phase === 'playing'; i += 1) game.update(DT);
    game.start();
    expect(game.phase).toBe('playing');
    expect(game.score.value).toBe(0);
    expect(game.road.speed).toBe(SPEED.start);
    expect(game.obstacles.items.length).toBe(0);
  });

  /**
   * A simple bot with human-like limits (150 ms reaction, one action per
   * 200 ms like the smoother's cooldown) must be able to survive. This is the
   * executable version of the "spawning is fair" claim.
   */
  it('is survivable by a latency-limited bot for 90 s on every seed', () => {
    const survived = (seed: number): number => {
      const game = new Game({ rng: createRng(seed) });
      game.start();
      const queue: { due: number; action: Gesture }[] = [];
      let now = 0;
      let lastExec = -1;
      while (game.phase === 'playing' && now < 90) {
        const decision = decide(game);
        if (decision !== null && queue.length === 0) queue.push({ due: now + 0.15, action: decision });
        const next = queue[0];
        if (next !== undefined && now >= next.due && now - lastExec >= 0.2) {
          game.apply(next.action);
          lastExec = now;
          queue.shift();
        }
        game.update(DT);
        now += DT;
      }
      return now;
    };
    const results = Array.from({ length: 20 }, (_, i) => survived(i + 1));
    const survivors = results.filter((t) => t >= 89.9).length;
    expect(survivors).toBe(20);
  });
});

function decide(game: Game): Gesture | null {
  const { car, road, obstacles } = game;
  const hazards = (lane: number) =>
    obstacles.items
      .filter((o) => o.lane === lane && o.y < CAR.y + CAR.height)
      .map((o) => ({ o, t: (CAR.y - (o.y + o.height)) / road.speed }))
      .filter((h) => h.t < 1.2)
      .sort((a, b) => a.t - b.t);
  const mine = hazards(car.lane)[0];
  if (mine === undefined) return null;
  if (mine.o.kind === 'barrier') return mine.t < 0.45 && !car.airborne ? 'JUMP' : null;
  if (mine.t > 0.8) return null;
  const clear = (lane: number): boolean => lane >= 0 && lane < ROAD.laneCount && hazards(lane).every((h) => h.o.kind === 'barrier');
  if (clear(car.lane - 1)) return 'LEFT';
  if (clear(car.lane + 1)) return 'RIGHT';
  return null;
}

describe('GameLoop', () => {
  function fakeScheduler(): { scheduler: FrameScheduler; step: (t: number) => void; pending: () => boolean } {
    let callback: ((t: number) => void) | null = null;
    return {
      scheduler: {
        request: (cb) => {
          callback = cb;
          return 1;
        },
        cancel: () => {
          callback = null;
        },
      },
      step: (t) => {
        const cb = callback;
        callback = null;
        cb?.(t);
      },
      pending: () => callback !== null,
    };
  }

  it('reports delta in seconds, starting at 0 and clamping long gaps', () => {
    const fake = fakeScheduler();
    const deltas: number[] = [];
    const loop = new GameLoop((d) => deltas.push(d), fake.scheduler);
    loop.start();
    fake.step(1000);
    fake.step(1016);
    fake.step(6000); // tab was hidden for 5 s
    expect(deltas[0]).toBe(0);
    expect(deltas[1]).toBeCloseTo(0.016, 5);
    expect(deltas[2]).toBe(0.05);
  });

  it('stops scheduling after stop() and ignores a double start', () => {
    const fake = fakeScheduler();
    const loop = new GameLoop(() => undefined, fake.scheduler);
    loop.start();
    loop.start();
    expect(loop.running).toBe(true);
    fake.step(0);
    expect(fake.pending()).toBe(true);
    loop.stop();
    expect(loop.running).toBe(false);
    expect(fake.pending()).toBe(false);
  });

  it('keeps OBSTACLES constants consistent with lane width', () => {
    expect(OBSTACLES.barrierWidth).toBeLessThan(ROAD.width / ROAD.laneCount);
  });
});
