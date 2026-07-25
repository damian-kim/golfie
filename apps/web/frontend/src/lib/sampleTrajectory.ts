import type { MetricSource, MetricValue, ScenePoint, TrajectoryPayload } from './types';

export interface SampleParameters { velocity: number; angle: number; direction: number; backspin: number; sidespin: number; drag: number; gravity: number }
export const DEFAULT_SAMPLE: SampleParameters = { velocity: 45, angle: 15, direction: 0, backspin: 3000, sidespin: 0, drag: .24, gravity: 9.81 };
const metric = (value: number | null, source: MetricSource = 'experimental'): MetricValue => ({ value, source, confidence: value === null ? 0 : 1, notes: 'Synthetic adjustable demo input.' });

function trajectory(p: SampleParameters): ScenePoint[] {
  const pitch = p.angle * Math.PI / 180; const yaw = p.direction * Math.PI / 180;
  let x = 0, y = 0, z = 0, vx = p.velocity * Math.cos(pitch) * Math.cos(yaw), vy = p.velocity * Math.sin(pitch), vz = p.velocity * Math.cos(pitch) * Math.sin(yaw), t = 0;
  const points: ScenePoint[] = [{ x, y, z, t, confidence: 1 }]; const dt = .02; const spinLift = Math.min(.36, Math.abs(p.backspin) / 15000); const curve = p.sidespin / 900000;
  while (t < 18) {
    const speed = Math.hypot(vx, vy, vz); const resistance = .5 * 1.225 * p.drag * .00143 * speed / .04593;
    vx += (-resistance * vx) * dt; vy += (-p.gravity - resistance * vy + spinLift * speed * .32) * dt; vz += (-resistance * vz + curve * speed * speed) * dt;
    x += vx * dt; y += vy * dt; z += vz * dt; t += dt;
    if (y <= 0 && t > .2) { y = 0; points.push({ x, y, z, t, confidence: 1 }); break; }
    points.push({ x, y, z, t, confidence: 1 });
  }
  return points;
}

export function makeSamplePayload(parameters: SampleParameters): TrajectoryPayload {
  const points = trajectory(parameters); const landing = points.at(-1)!; const apex = Math.max(...points.map((point) => point.y));
  return { session_id: 'sample', club: 'Driver', is_placeholder: true, warnings: [], notes: 'Adjustable synthetic trajectory. No swing footage is attached.', measured_points: [], fitted_points: [], simulated_trajectory: points, metrics: { ball_speed_mps: metric(parameters.velocity), launch_angle_deg: metric(parameters.angle), horizontal_launch_deg: metric(parameters.direction), carry_m: metric(landing.x), total_m: metric(landing.x), apex_m: metric(apex), side_deviation_m: metric(landing.z), backspin_rpm: metric(parameters.backspin), sidespin_rpm: metric(parameters.sidespin), spin_axis_deg: metric(null, 'not_available'), club_speed_mps: metric(null, 'not_available'), smash_factor: metric(null, 'not_available') } };
}
