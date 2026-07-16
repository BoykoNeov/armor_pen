extends MultiMeshInstance2D
## Plays back one baked cache as a moving 2D point cloud — one draw call for the
## whole set (MultiMeshInstance2D), colored per particle. Knows ONLY the cache
## format (CLAUDE.md §2): no solver, no physics, no Godot-side simulation.
##
## Column layout comes from the manifest via CacheLoader; offsets are never
## hardcoded (CACHE_FORMAT §2).
##
## Run interactively:
##   godot --path visualizer -- --cache M:/path/to/caches/apfsds_vs_era
## Capture proof frames (renders a spread of frames to PNG, then quits):
##   godot --path visualizer -- --cache <dir> --shots M:/path/to/out_dir
## With no --cache it plays the golden fixture, so the viewer runs with no
## solver present.
##
## Rendering notes (all presentation-side; the cache is untouched):
## - Particles are soft discs (shaders/particle.gdshader). Per-instance
##   custom data carries a "heat" scalar: damaged/fast particles render
##   larger and HDR-bright, and a WorldEnvironment glow pass blooms them.
## - Free fragments (damage > 0.5) are stretched along their motion vector
##   (previous frame vs this frame) so spall reads as streaking sparks.
## - Detonated ERA filler renders as an incandescent fireball rather than
##   flat white.

## Cache directory to play. A res:// path or an absolute OS path both work
## (CacheLoader uses FileAccess, which handles either). Overridden by --cache.
@export var cache_dir: String = "res://fixtures/tiny_golden_cache"

## How to color particles: "material_id" (discrete palette) or a scalar
## attribute name like "vel_mag" / "stress" / "damage" (inferno ramp).
@export var color_by: String = "material_id"

## World-space edge length of each particle quad (domain units, e.g. mm).
@export var point_size: float = 1.0

## Simulated playback rate: how many baked frames to advance per wall second.
##
## Default 10, not 24, because THIS is what decides smoothness — not the number of
## frames in the cache. _show_frame costs ~100 ms at 137k particles and ~220 ms at
## 287k, so above ~10 fps (and ~4.5 on the heaviest decks) the viewer cannot draw
## every baked frame and starts skipping them — which discards exactly the extra
## resolution the finer frame_dt was baked for. A lower rate plays every frame:
## slower in wall-clock, but actually smooth. ↑/↓ retune it per deck.
@export var frames_per_second: float = 10.0

## Fraction of the viewport the domain fills (1.0 = edge to edge).
@export var fit_margin: float = 0.92

# Discrete colors per material id (name lookup is via the manifest's `materials`).
const MATERIAL_COLORS := {
	0: Color(0.98, 0.78, 0.24),  # tungsten_rod — bright gold
	1: Color(0.43, 0.50, 0.61),  # rha — steel blue-gray
	2: Color(0.82, 0.72, 0.48),  # ceramic — tan
	3: Color(0.95, 0.38, 0.12),  # era_filler — hot orange
	4: Color(0.52, 0.26, 0.18),  # era_filler_inert — dark brick
	5: Color(0.64, 0.42, 0.80),  # nera_filler — violet (never ignites, never spalls,
	#                              so it keeps this base tone for the whole bake —
	#                              deliberately far from rha's gray-blue so the
	#                              cohesive interlayer reads against the plates)
}
const FALLBACK_COLOR := Color(0.6, 0.6, 0.6)
# Damaged/spalled particles trend toward this hot-spark tone; "heat" (glow)
# scales with damage and speed so fresh fast fragments burn brightest.
const SPARK_COLOR := Color(1.0, 0.84, 0.58)
# Speed (domain units/s ~ m/s) at which a fragment reaches full glow.
const SPARK_FULL_GLOW_SPEED := 900.0
const ACCENT := Color(0.95, 0.75, 0.30)

const CacheLoaderScript = preload("res://scripts/cache_loader.gd")
const PARTICLE_SHADER = preload("res://shaders/particle.gdshader")

var _loader := CacheLoaderScript.new()
var _frame: int = 0
var _accum: float = 0.0
var _playing: bool = true
var _shots_dir: String = ""

var _pos_x: int = -1
var _pos_y: int = -1
var _vel_col: int = -1
var _mat_col: int = -1
var _damage_col: int = -1
var _color_col: int = -1
var _color_hi := {}          # attribute name -> normalization max (cached per mode)
var _color_span: float = 1.0

# Deterministic per-particle brightness jitter so solid bodies read as grainy
# metal instead of a flat fill.
var _jitter := PackedFloat32Array()

# Previous frame's raw data, for motion streaks. During sequential playback the
# shown frame becomes next frame's "previous" — no extra disk reads.
var _prev_data := PackedFloat32Array()
var _prev_frame: int = -2

# Zoom factor per wheel notch, and the zoom range allowed relative to the
# fit-the-whole-domain baseline (0.25x = pull back to see the field around the
# domain; 40x = down to individual particles).
const ZOOM_STEP := 1.15
const MIN_ZOOM_REL := 0.25
const MAX_ZOOM_REL := 40.0

var _camera: Camera2D
var _fit_zoom: float = 1.0     # zoom that frames the whole domain (the F-key reset)
var _fit_center := Vector2.ZERO
var _hud_info: Label
var _legend_box: HBoxContainer
var _timeline_fill: ColorRect

# Color modes reachable with the C key: material_id plus every scalar
# attribute in the manifest that isn't a position.
var _color_modes: PackedStringArray = PackedStringArray()


func _ready() -> void:
	_apply_cmdline_overrides()

	var err = _loader.load_cache(cache_dir)
	if err != OK:
		push_error("cache load failed for '%s': %s" % [cache_dir, str(err)])
		get_tree().quit(1)
		return

	_pos_x = _loader.attribute_index("pos_x")
	_pos_y = _loader.attribute_index("pos_y")
	_vel_col = _loader.attribute_index("vel_mag")
	_mat_col = _loader.attribute_index("material_id")
	_damage_col = _loader.attribute_index("damage")
	if _pos_x < 0 or _pos_y < 0:
		push_error("cache lacks pos_x/pos_y — cannot draw")
		get_tree().quit(1)
		return

	_collect_color_modes()

	multimesh = MultiMesh.new()
	multimesh.transform_format = MultiMesh.TRANSFORM_2D
	multimesh.use_colors = true
	multimesh.use_custom_data = true
	multimesh.mesh = _make_point_mesh(point_size)
	multimesh.instance_count = _loader.particle_count

	var mat := ShaderMaterial.new()
	mat.shader = PARTICLE_SHADER
	material = mat

	_jitter.resize(_loader.particle_count)
	for p in _loader.particle_count:
		_jitter[p] = 0.86 + 0.28 * fposmod(sin(float(p) * 12.9898) * 43758.5453, 1.0)

	_setup_environment()
	_setup_background()
	_setup_camera()
	_setup_hud()
	_set_color_mode(color_by)
	_show_frame(0)

	if _shots_dir != "":
		set_process(false)
		_run_capture()   # captures a spread of frames to PNG, then quits


func _process(delta: float) -> void:
	if not _playing:
		return
	_accum += delta * frames_per_second
	if _accum < 1.0:
		return
	# Advance straight to the target frame in ONE _show_frame call.
	#
	# This used to loop `while _accum >= 1.0`, drawing every intermediate frame.
	# _show_frame costs ~100 ms at 137k particles and ~220 ms at 287k (measured —
	# see the capture path), so whenever it exceeds 1/frames_per_second that loop
	# rendered N frames nobody saw, which made the next `delta` N times larger,
	# which rendered N² more: an unbounded catch-up spiral, and every one of those
	# frames also re-read its slice of frames.bin off disk.
	#
	# Skipping to the target bounds the work at one draw per _process. Note it
	# still SKIPS baked frames whenever the renderer can't keep up, which throws
	# away the resolution the extra frames bought — press ↓ to lower
	# frames_per_second until playback stops skipping. The real fix is making
	# _show_frame cheaper (it runs a per-particle GDScript loop with three
	# set_instance_* calls each; MultiMesh.set_buffer would collapse that to one).
	var advance := int(_accum)
	_accum -= float(advance)
	_frame = (_frame + advance) % _loader.frame_count
	_show_frame(_frame)


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseButton and event.pressed:
		match event.button_index:
			MOUSE_BUTTON_WHEEL_UP:
				_zoom_at_cursor(ZOOM_STEP)
				return
			MOUSE_BUTTON_WHEEL_DOWN:
				_zoom_at_cursor(1.0 / ZOOM_STEP)
				return
	# Drag with middle or right button to pan. `relative` is in screen pixels, so
	# dividing by zoom converts it to world units — the grab point stays under the
	# cursor at any zoom level.
	if event is InputEventMouseMotion:
		# `int(...)`: button_mask is a BitField, whose `&` result GDScript cannot
		# infer a type for, so `:=` is a parse error here.
		var dragging: int = int(event.button_mask) & int(
			MOUSE_BUTTON_MASK_MIDDLE | MOUSE_BUTTON_MASK_RIGHT)
		if dragging != 0:
			_camera.position -= event.relative / _camera.zoom
			_update_hud()
		return

	if not (event is InputEventKey) or not event.pressed:
		return
	match event.keycode:
		KEY_EQUAL, KEY_KP_ADD:
			_zoom_at_cursor(ZOOM_STEP)
		KEY_MINUS, KEY_KP_SUBTRACT:
			_zoom_at_cursor(1.0 / ZOOM_STEP)
		KEY_F:
			_reset_view()
		KEY_SPACE:
			_playing = not _playing
			_update_hud()
		KEY_R:
			_frame = 0
			_show_frame(_frame)
		KEY_RIGHT:
			_playing = false
			_frame = (_frame + 1) % _loader.frame_count
			_show_frame(_frame)
		KEY_LEFT:
			_playing = false
			_frame = (_frame - 1 + _loader.frame_count) % _loader.frame_count
			_show_frame(_frame)
		KEY_UP:
			frames_per_second = minf(frames_per_second * 1.5, 240.0)
			_update_hud()
		KEY_DOWN:
			frames_per_second = maxf(frames_per_second / 1.5, 1.0)
			_update_hud()
		KEY_C:
			var i := (_color_modes.find(color_by) + 1) % _color_modes.size()
			_set_color_mode(_color_modes[i])
			_show_frame(_frame)
		KEY_ESCAPE:
			get_tree().quit(0)


# --- frame drawing -----------------------------------------------------------

func _show_frame(f: int) -> void:
	_frame = f
	var data := _loader.read_frame(f)
	var prev := PackedFloat32Array()
	if f > 0:
		prev = _prev_data if _prev_frame == f - 1 else _loader.read_frame(f - 1)

	var stride := _loader.attributes.size()
	var ymax := float(_loader.domain.get("ymax", 0.0))
	var ymin := float(_loader.domain.get("ymin", 0.0))
	var use_material := color_by == "material_id" and _mat_col >= 0
	var span := _color_span
	var has_prev := prev.size() == data.size()
	var streak_min := point_size * 0.7

	for p in _loader.particle_count:
		var base := p * stride
		var x := data[base + _pos_x]
		# Flip Y (physics is y-up; Godot 2D is y-down) by mirroring inside the
		# domain so the camera framing below stays simple.
		var wy := ymin + ymax - data[base + _pos_y]

		var dmg := 0.0
		if _damage_col >= 0:
			dmg = clampf(data[base + _damage_col], 0.0, 1.0)
		var speed := data[base + _vel_col] if _vel_col >= 0 else 0.0

		# Free fragments streak along their motion vector; everything else is
		# an unrotated disc.
		var xform: Transform2D
		if has_prev and dmg > 0.5:
			var dx := x - prev[base + _pos_x]
			var dy := prev[base + _pos_y] - data[base + _pos_y]  # flipped
			var len2 := dx * dx + dy * dy
			if len2 > streak_min * streak_min:
				var l := sqrt(len2)
				var sx := clampf(1.0 + l / (point_size * 2.0), 1.0, 5.0)
				var sy := clampf(1.0 / sqrt(sx), 0.55, 1.0)
				xform = Transform2D(atan2(dy, dx), Vector2(sx, sy), 0.0, Vector2(x, wy))
			else:
				xform = Transform2D(0.0, Vector2(x, wy))
		else:
			xform = Transform2D(0.0, Vector2(x, wy))
		multimesh.set_instance_transform_2d(p, xform)

		var col: Color
		var heat := 0.0
		var mid := int(round(data[base + _mat_col])) if _mat_col >= 0 else -1
		if use_material:
			col = MATERIAL_COLORS.get(mid, FALLBACK_COLOR)
		else:
			var t := clampf(data[base + _color_col] / span, 0.0, 1.0)
			col = _inferno(t)
			heat = maxf(0.0, t - 0.75) * 2.4

		if dmg > 0.5:
			# Detached fragment: glow scales with damage x speed, so fresh fast
			# fragments burn bright while settled debris cools toward its base color.
			var s := (dmg - 0.5) * 2.0
			var v := clampf(speed / SPARK_FULL_GLOW_SPEED, 0.0, 1.0)
			if use_material:
				if mid == 3:  # detonated ERA filler: orange fireball, not pale spall
					col = col.lerp(Color(1.0, 0.60, 0.20), s)
				else:
					col = col.lerp(SPARK_COLOR, s * (0.35 + 0.45 * v))
			heat = maxf(heat, s * (0.10 + 0.90 * v))
			if mid == 3:
				heat = minf(heat + 0.15, 1.0)

		var j := _jitter[p]
		multimesh.set_instance_color(p, Color(col.r * j, col.g * j, col.b * j))
		multimesh.set_instance_custom_data(p, Color(heat, 0.0, 0.0, 0.0))

	_prev_data = data
	_prev_frame = f
	_update_hud()


## Polynomial fit of matplotlib's inferno colormap (public shadertoy fit by
## Matt Zucker) — perceptually uniform, reads well on the dark background.
func _inferno(t: float) -> Color:
	var c0 := Vector3(0.0002189403691192265, 0.001651004631001012, -0.01948089843709184)
	var c1 := Vector3(0.1065134194856116, 0.5639564367884091, 3.932712388889277)
	var c2 := Vector3(11.60249308247187, -3.972853965665698, -15.9423941062914)
	var c3 := Vector3(-41.70399613139459, 17.43639888205313, 44.35414519872813)
	var c4 := Vector3(77.162935699427, -33.40235894210092, -81.80730925738993)
	var c5 := Vector3(-71.31942824499214, 32.62606426397723, 73.20951985803202)
	var c6 := Vector3(25.13112622477341, -12.24266895238567, -23.07032500287172)
	var v := c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))
	return Color(clampf(v.x, 0.0, 1.0), clampf(v.y, 0.0, 1.0), clampf(v.z, 0.0, 1.0))


# --- color modes -------------------------------------------------------------

func _collect_color_modes() -> void:
	_color_modes.clear()
	if _mat_col >= 0:
		_color_modes.append("material_id")
	for a in _loader.attributes:
		if a not in ["pos_x", "pos_y", "material_id"]:
			_color_modes.append(a)
	if _color_modes.is_empty():
		_color_modes.append("material_id")


func _set_color_mode(mode: String) -> void:
	if mode != "material_id" and _loader.attribute_index(mode) < 0:
		push_warning("unknown color attribute '%s', using material_id" % mode)
		mode = "material_id"
	color_by = mode
	_color_col = _loader.attribute_index(mode)
	if mode != "material_id" and _color_col >= 0:
		if not _color_hi.has(mode):
			_color_hi[mode] = _scan_color_range(_color_col)
		_color_span = maxf(_color_hi[mode], 1e-6)
	_rebuild_legend()
	_update_hud()


## Normalization max for a scalar attribute: 99.5th percentile of a subsample
## across three frames, so a single outlier particle doesn't wash out the ramp.
func _scan_color_range(col: int) -> float:
	var stride := _loader.attributes.size()
	var samples := PackedFloat32Array()
	for f in [_loader.frame_count / 4, _loader.frame_count / 2, (3 * _loader.frame_count) / 4]:
		var data := _loader.read_frame(f)
		var p := 0
		while p < _loader.particle_count:
			samples.append(data[p * stride + col])
			p += 8
	var arr := Array(samples)
	arr.sort()
	return float(arr[int(float(arr.size() - 1) * 0.995)])


# --- setup helpers -----------------------------------------------------------

func _apply_cmdline_overrides() -> void:
	var user_args := OS.get_cmdline_user_args()
	for i in user_args.size():
		match user_args[i]:
			"--cache":
				if i + 1 < user_args.size():
					cache_dir = user_args[i + 1]
			"--shots":
				if i + 1 < user_args.size():
					_shots_dir = user_args[i + 1]
			"--color":
				if i + 1 < user_args.size():
					color_by = user_args[i + 1]


func _setup_environment() -> void:
	# 2D glow: HDR colors from the particle shader bloom into halos.
	var env := Environment.new()
	env.background_mode = Environment.BG_CLEAR_COLOR
	env.glow_enabled = true
	env.glow_blend_mode = Environment.GLOW_BLEND_MODE_ADDITIVE
	env.glow_intensity = 0.35
	env.glow_strength = 1.0
	env.glow_bloom = 0.02
	env.glow_hdr_threshold = 1.1
	var we := WorldEnvironment.new()
	we.environment = env
	add_child(we)


func _setup_background() -> void:
	# Vignetted gradient backdrop on a far canvas layer, plus a faint domain
	# frame + grid in world space so the playback has spatial context.
	var layer := CanvasLayer.new()
	layer.layer = -10
	add_child(layer)
	var rect := ColorRect.new()
	rect.set_anchors_preset(Control.PRESET_FULL_RECT)
	var sh := Shader.new()
	sh.code = """
shader_type canvas_item;
void fragment() {
	vec3 top = vec3(0.085, 0.095, 0.125);
	vec3 bot = vec3(0.035, 0.040, 0.055);
	vec3 col = mix(top, bot, UV.y);
	float d = length(UV - vec2(0.5, 0.45));
	col *= 1.0 - 0.5 * smoothstep(0.35, 0.95, d);
	COLOR = vec4(col, 1.0);
}
"""
	var mat := ShaderMaterial.new()
	mat.shader = sh
	rect.material = mat
	layer.add_child(rect)

	var grid := Node2D.new()
	grid.z_index = -1
	grid.draw.connect(_draw_domain.bind(grid))
	add_child(grid)


func _draw_domain(node: Node2D) -> void:
	var xmin := float(_loader.domain.get("xmin", 0.0))
	var xmax := float(_loader.domain.get("xmax", 100.0))
	var ymin := float(_loader.domain.get("ymin", 0.0))
	var ymax := float(_loader.domain.get("ymax", 100.0))
	var r := Rect2(xmin, ymin, xmax - xmin, ymax - ymin)
	node.draw_rect(r, Color(1, 1, 1, 0.018), true)
	var step := 25.0
	var x := xmin + step
	while x < xmax - 1e-6:
		node.draw_line(Vector2(x, ymin), Vector2(x, ymax), Color(1, 1, 1, 0.035), 0.15)
		x += step
	var y := ymin + step
	while y < ymax - 1e-6:
		node.draw_line(Vector2(xmin, y), Vector2(xmax, y), Color(1, 1, 1, 0.035), 0.15)
		y += step
	node.draw_rect(r, Color(1, 1, 1, 0.16), false, 0.3)


func _setup_camera() -> void:
	_camera = Camera2D.new()
	add_child(_camera)
	var xmin := float(_loader.domain.get("xmin", 0.0))
	var xmax := float(_loader.domain.get("xmax", 100.0))
	var ymin := float(_loader.domain.get("ymin", 0.0))
	var ymax := float(_loader.domain.get("ymax", 100.0))
	var dom_w := maxf(xmax - xmin, 1e-6)
	var dom_h := maxf(ymax - ymin, 1e-6)
	var vp := get_viewport().get_visible_rect().size
	_fit_zoom = minf(vp.x / dom_w, vp.y / dom_h) * fit_margin
	_fit_center = Vector2((xmin + xmax) * 0.5, (ymin + ymax) * 0.5)
	_camera.make_current()
	_reset_view()


## Zoom about the cursor: the world point under the mouse stays pinned there, so
## you can drill into the crater without it sliding out of view.
##
## The camera offset is solved directly rather than by reading
## get_global_mouse_position() before and after: the viewport's canvas transform
## does not necessarily refresh within the same frame we write _camera.zoom, so
## the "after" read can still be at the old zoom and the pivot drifts. With
## anchor_mode DRAG_CENTER, world = position + (screen - viewport_centre) / zoom;
## holding the world point fixed across a zoom change gives the shift below.
func _zoom_at_cursor(factor: float) -> void:
	var z0 := _camera.zoom.x
	var z1 := clampf(z0 * factor, _fit_zoom * MIN_ZOOM_REL, _fit_zoom * MAX_ZOOM_REL)
	if is_equal_approx(z0, z1):
		return    # already at a zoom limit
	var screen_off := get_viewport().get_mouse_position() \
		- get_viewport().get_visible_rect().size * 0.5
	_camera.position += screen_off * (1.0 / z0 - 1.0 / z1)
	_camera.zoom = Vector2(z1, z1)
	_update_hud()


## Back to the framing the viewer opens with: whole domain, centred.
func _reset_view() -> void:
	_camera.zoom = Vector2(_fit_zoom, _fit_zoom)
	_camera.position = _fit_center
	_update_hud()


func _setup_hud() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)

	var panel := PanelContainer.new()
	panel.position = Vector2(12, 10)
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.04, 0.05, 0.08, 0.88)
	style.set_corner_radius_all(6)
	style.set_content_margin_all(10)
	panel.add_theme_stylebox_override("panel", style)
	layer.add_child(panel)

	var box := VBoxContainer.new()
	box.add_theme_constant_override("separation", 4)
	panel.add_child(box)

	var title := Label.new()
	title.text = cache_dir.get_file()
	title.add_theme_font_size_override("font_size", 17)
	title.add_theme_color_override("font_color", Color(0.95, 0.93, 0.88))
	box.add_child(title)

	_hud_info = Label.new()
	_hud_info.add_theme_font_size_override("font_size", 13)
	_hud_info.add_theme_color_override("font_color", Color(0.75, 0.78, 0.84))
	box.add_child(_hud_info)

	_legend_box = HBoxContainer.new()
	_legend_box.add_theme_constant_override("separation", 10)
	box.add_child(_legend_box)

	var keys := Label.new()
	keys.text = "space play/pause   ←/→ step   ↑/↓ speed   C color   R restart   " \
		+ "wheel/±  zoom   drag pan   F fit   Esc quit"
	keys.add_theme_font_size_override("font_size", 11)
	keys.add_theme_color_override("font_color", Color(0.55, 0.58, 0.64))
	box.add_child(keys)

	# Timeline bar along the bottom edge.
	var track := ColorRect.new()
	track.color = Color(1, 1, 1, 0.10)
	track.set_anchors_preset(Control.PRESET_BOTTOM_WIDE)
	track.offset_top = -4.0
	layer.add_child(track)
	_timeline_fill = ColorRect.new()
	_timeline_fill.color = ACCENT
	_timeline_fill.set_anchors_preset(Control.PRESET_BOTTOM_WIDE)
	_timeline_fill.offset_top = -4.0
	_timeline_fill.anchor_right = 0.0
	layer.add_child(_timeline_fill)

	_update_hud()


func _rebuild_legend() -> void:
	if _legend_box == null:
		return
	for c in _legend_box.get_children():
		c.queue_free()
	if color_by == "material_id":
		for key in _loader.materials:
			var mid := int(key)
			_legend_box.add_child(_swatch(MATERIAL_COLORS.get(mid, FALLBACK_COLOR),
				String(_loader.materials[key])))
		_legend_box.add_child(_swatch(SPARK_COLOR, "spall"))
	else:
		var grad := Gradient.new()
		grad.offsets = PackedFloat32Array([0.0, 0.25, 0.5, 0.75, 1.0])
		grad.colors = PackedColorArray([_inferno(0.0), _inferno(0.25), _inferno(0.5),
			_inferno(0.75), _inferno(1.0)])
		var tex := GradientTexture1D.new()
		tex.gradient = grad
		var bar := TextureRect.new()
		bar.texture = tex
		bar.custom_minimum_size = Vector2(140, 10)
		bar.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
		bar.stretch_mode = TextureRect.STRETCH_SCALE
		bar.size_flags_vertical = Control.SIZE_SHRINK_CENTER
		var lo := Label.new()
		lo.text = "0"
		lo.add_theme_font_size_override("font_size", 11)
		lo.add_theme_color_override("font_color", Color(0.7, 0.72, 0.78))
		var hi := Label.new()
		hi.text = "%s  (%s)" % [String.num(_color_span, 0 if _color_span >= 100.0 else 2), color_by]
		hi.add_theme_font_size_override("font_size", 11)
		hi.add_theme_color_override("font_color", Color(0.7, 0.72, 0.78))
		_legend_box.add_child(lo)
		_legend_box.add_child(bar)
		_legend_box.add_child(hi)


func _swatch(col: Color, label_text: String) -> HBoxContainer:
	var h := HBoxContainer.new()
	h.add_theme_constant_override("separation", 4)
	var sq := ColorRect.new()
	sq.color = col
	sq.custom_minimum_size = Vector2(11, 11)
	sq.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	h.add_child(sq)
	var l := Label.new()
	l.text = label_text
	l.add_theme_font_size_override("font_size", 12)
	l.add_theme_color_override("font_color", Color(0.78, 0.80, 0.85))
	h.add_child(l)
	return h


func _update_hud() -> void:
	if _hud_info == null:
		return
	var t_ms := _frame * _loader.frame_dt * 1000.0
	# Zoom reads as a percentage of the fit-the-domain baseline, so 100% is
	# always "the whole field" no matter how big the scenario's domain is.
	var zoom_pct := 100.0 if _camera == null else _camera.zoom.x / _fit_zoom * 100.0
	_hud_info.text = "frame %d / %d    t = %.3f ms    %.0f fps    zoom %.0f%%%s" % [
		_frame, _loader.frame_count - 1, t_ms, frames_per_second, zoom_pct,
		"" if _playing else "    ⏸ paused",
	]
	if _timeline_fill != null and _loader.frame_count > 1:
		_timeline_fill.anchor_right = float(_frame) / float(_loader.frame_count - 1)


func _make_point_mesh(size: float) -> QuadMesh:
	var q := QuadMesh.new()
	q.size = Vector2(size, size)
	return q


# --- capture (my verification path; produces PNGs of the bake in motion) -----

func _run_capture() -> void:
	var dir := _shots_dir
	DirAccess.make_dir_recursive_absolute(dir)
	var n := _loader.frame_count
	var targets := [0, n / 4, n / 2, (3 * n) / 4, n - 1]
	for f in targets:
		var t0 := Time.get_ticks_usec()
		_show_frame(f)
		var cost_ms := float(Time.get_ticks_usec() - t0) / 1000.0
		await RenderingServer.frame_post_draw
		await RenderingServer.frame_post_draw
		var img := get_viewport().get_texture().get_image()
		var path := dir.path_join("frame_%03d.png" % f)
		var err := img.save_png(path)
		# _show_frame cost is the playback budget: it runs a per-particle GDScript
		# loop plus a frame read, once per baked frame. If it exceeds
		# 1000/frames_per_second ms, _process's catch-up loop starts re-reading and
		# re-rendering frames nobody sees, and playback skips. Printing it here is
		# the only cheap measurement available — capture sets process off, so this
		# is the one code path that exercises _show_frame outside interactive play.
		print("SHOT frame %d -> %s (%d)  _show_frame=%.1f ms (budget %.1f ms @ %.0f fps)"
			% [f, path, err, cost_ms, 1000.0 / frames_per_second, frames_per_second])
	get_tree().quit(0)
