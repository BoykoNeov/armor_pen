// Soft-disc particle shader for the playback MultiMesh.
// Per-instance COLOR is the material/scalar color chosen by particle_view.gd;
// INSTANCE_CUSTOM.x is "heat" in [0,1] — hot particles render larger, brighter
// (HDR, feeds the 2D glow pass) and with a wider halo.
shader_type canvas_item;
render_mode unshaded;

varying float v_heat;

void vertex() {
	v_heat = INSTANCE_CUSTOM.x;
	// Hot fragments grow so the spark/fireball reads at domain scale.
	VERTEX *= 1.0 + v_heat * 0.7;
}

void fragment() {
	vec2 d = (UV - vec2(0.5)) * 2.0;
	float r = length(d);
	// Soft disc: solid core, feathered edge. Heat widens the feather into a halo.
	float edge_in = mix(0.55, 0.15, v_heat);
	float body = smoothstep(1.0, edge_in, r);
	// Slight center highlight gives dense solids a granular, lit look
	// instead of a flat fill.
	float shade = mix(0.80, 1.18, smoothstep(1.0, 0.0, r));
	vec3 col = COLOR.rgb * shade;
	// Incandescent HDR boost — values above 1.0 are picked up by glow.
	col *= 1.0 + v_heat * v_heat * 2.2;
	COLOR = vec4(col, COLOR.a * body);
}
