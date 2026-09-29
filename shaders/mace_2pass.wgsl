// MaCE two-pass, torus wrap. 2D 9-neighborhood.
// Pass A writes Z[x] = sum_{y in N(x)} exp(beta * A[y])
// Pass B writes rho'[x] = sum_{y in N(x)} rho[y] * exp(beta * A[x]) / Z[y]
//
// Bindings assumed:
//   params.field_w, field_h, transport_beta
//   affinity, rho_in, denom, rho_out : array<f32>  (single channel shown)

fn wrap_idx(x: i32, n: i32) -> i32 {
  return (x % n + n) % n;
}

fn cell(x: i32, y: i32, w: i32, h: i32) -> u32 {
  return u32(wrap_idx(y, h) * w + wrap_idx(x, w));
}

@compute @workgroup_size(64)
fn mace_denom(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = i32(params.field_w);
  let h = i32(params.field_h);
  let n = u32(w * h);
  if (gid.x >= n) { return; }
  let x = i32(gid.x) % w;
  let y = i32(gid.x) / w;
  var z = 0.0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      z += exp(params.transport_beta * affinity[cell(x + dx, y + dy, w, h)]);
    }
  }
  denom[gid.x] = z;
}

@compute @workgroup_size(64)
fn mace_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = i32(params.field_w);
  let h = i32(params.field_h);
  let n = u32(w * h);
  if (gid.x >= n) { return; }
  let x = i32(gid.x) % w;
  let y = i32(gid.x) / w;
  let w_here = exp(params.transport_beta * affinity[gid.x]);
  var acc = 0.0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let j = cell(x + dx, y + dy, w, h);
      acc += rho_in[j] * w_here / denom[j];
    }
  }
  rho_out[gid.x] = acc;
}
