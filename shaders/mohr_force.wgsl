// Corrected Mohr tent. F is dimensionless. Do not multiply by r_max.
// accel += gain * F * normalize(delta)

fn mohr_force(r: f32, a: f32, beta: f32) -> f32 {
  if (r < beta) {
    return r / beta - 1.0;
  }
  if (r <= 1.0) {
    return a * (1.0 - abs(2.0 * r - 1.0 - beta) / (1.0 - beta));
  }
  return 0.0;
}

fn wrapped_delta_2d(a: vec2<f32>, b: vec2<f32>, world: vec2<f32>) -> vec2<f32> {
  var d = a - b;
  d = d - world * round(d / world);
  return d;
}

// Fragment of the force loop body (2D):
// let delta = wrapped_delta_2d(pj.xy, pos_i, world);
// let dist = length(delta);
// if (dist < 1e-6) { continue; }
// let r = dist / params.r_max;
// if (r > 1.0) { continue; }
// let a = matrix[type_i * stride + u32(pj.w)];
// let f = mohr_force(r, a, params.beta);
// accel += params.gain * f * normalize(delta);
//
// Friction in integrate:
// let decay = exp(-params.lambda * params.dt);
// vel = vel * decay + acc * params.dt;
