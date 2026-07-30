/**
 * vec3.js
 *
 * Minimal 3D vector math — just what the force-field minimizer needs.
 */

window.CC = window.CC || {};

CC.vec3 = {
  add: function (a, b) { return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z }; },
  sub: function (a, b) { return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z }; },
  scale: function (a, s) { return { x: a.x * s, y: a.y * s, z: a.z * s }; },
  dot: function (a, b) { return a.x * b.x + a.y * b.y + a.z * b.z; },
  cross: function (a, b) {
    return {
      x: a.y * b.z - a.z * b.y,
      y: a.z * b.x - a.x * b.z,
      z: a.x * b.y - a.y * b.x,
    };
  },
  length: function (a) { return Math.sqrt(CC.vec3.dot(a, a)); },
  normalize: function (a) {
    const len = CC.vec3.length(a);
    if (len < 1e-9) return { x: 1, y: 0, z: 0 };
    return CC.vec3.scale(a, 1 / len);
  },
  distance: function (a, b) { return CC.vec3.length(CC.vec3.sub(a, b)); },

  // Rotate point `p` by `angle` radians around the axis through `origin`
  // in direction `axis` (need not be pre-normalized). Rodrigues' rotation
  // formula -- used for torsion-driven conformer seeding (rotating one
  // side of a rotatable bond) and could equally be reused anywhere else
  // an arbitrary-axis rotation is needed.
  rotateAroundAxis: function (p, origin, axis, angle) {
    const k = CC.vec3.normalize(axis);
    const v = CC.vec3.sub(p, origin);
    const cosT = Math.cos(angle);
    const sinT = Math.sin(angle);
    const term1 = CC.vec3.scale(v, cosT);
    const term2 = CC.vec3.scale(CC.vec3.cross(k, v), sinT);
    const term3 = CC.vec3.scale(k, CC.vec3.dot(k, v) * (1 - cosT));
    const rotated = CC.vec3.add(CC.vec3.add(term1, term2), term3);
    return CC.vec3.add(origin, rotated);
  },
};
