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

## Cache directory to play. A res:// path or an absolute OS path both work
## (CacheLoader uses FileAccess, which handles either). Overridden by --cache.
@export var cache_dir: String = "res://fixtures/tiny_golden_cache"

## How to color particles: "material_id" (discrete palette) or a scalar
## attribute name like "vel_mag" / "stress" / "damage" (normalized ramp).
@export var color_by: String = "material_id"

## World-space edge length of each particle quad (domain units, e.g. mm).
@export var point_size: float = 0.8

## Simulated playback rate: how many baked frames to advance per wall second.
@export var frames_per_second: float = 24.0

## Fraction of the viewport the domain fills (1.0 = edge to edge).
@export var fit_margin: float = 0.92

# Discrete colors per material id (name lookup is via the manifest's `materials`).
const MATERIAL_COLORS := {
	0: Color(1.00, 0.80, 0.20),  # tungsten_rod — bright gold
	1: Color(0.42, 0.48, 0.58),  # rha — steel blue-gray
	2: Color(0.80, 0.70, 0.45),  # ceramic — tan
	3: Color(0.95, 0.35, 0.15),  # era_filler — hot orange
	4: Color(0.45, 0.20, 0.15),  # era_filler_inert — dark brick
}
const FALLBACK_COLOR := Color(0.6, 0.6, 0.6)
# A damaged/spalled particle is whitened toward this, so the fragment spray reads.
const SPALL_COLOR := Color(0.98, 0.98, 0.98)

const CacheLoaderScript = preload("res://scripts/cache_loader.gd")

var _loader := CacheLoaderScript.new()
var _frame: int = 0
var _accum: float = 0.0
var _playing: bool = true
var _shots_dir: String = ""

var _pos_x: int = -1
var _pos_y: int = -1
var _mat_col: int = -1
var _damage_col: int = -1
var _color_col: int = -1
var _color_lo: float = 0.0
var _color_hi: float = 1.0

var _camera: Camera2D
var _hud: Label


func _ready() -> void:
	_apply_cmdline_overrides()

	var err = _loader.load_cache(cache_dir)
	if err != OK:
		push_error("cache load failed for '%s': %s" % [cache_dir, str(err)])
		get_tree().quit(1)
		return

	_pos_x = _loader.attribute_index("pos_x")
	_pos_y = _loader.attribute_index("pos_y")
	_mat_col = _loader.attribute_index("material_id")
	_damage_col = _loader.attribute_index("damage")
	_color_col = _loader.attribute_index(color_by)
	if _pos_x < 0 or _pos_y < 0:
		push_error("cache lacks pos_x/pos_y — cannot draw")
		get_tree().quit(1)
		return
	_compute_color_range()

	multimesh = MultiMesh.new()
	multimesh.transform_format = MultiMesh.TRANSFORM_2D
	multimesh.use_colors = true
	multimesh.mesh = _make_point_mesh(point_size)
	multimesh.instance_count = _loader.particle_count

	_setup_camera()
	_setup_hud()
	_show_frame(0)

	if _shots_dir != "":
		set_process(false)
		_run_capture()   # captures a spread of frames to PNG, then quits


func _process(delta: float) -> void:
	if not _playing:
		return
	_accum += delta * frames_per_second
	while _accum >= 1.0:
		_accum -= 1.0
		_frame = (_frame + 1) % _loader.frame_count
		_show_frame(_frame)


func _unhandled_input(event: InputEvent) -> void:
	if not (event is InputEventKey) or not event.pressed:
		return
	match event.keycode:
		KEY_SPACE:
			_playing = not _playing
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
		KEY_ESCAPE:
			get_tree().quit(0)


# --- frame drawing -----------------------------------------------------------

func _show_frame(f: int) -> void:
	_frame = f
	var data := _loader.read_frame(f)
	var stride := _loader.attributes.size()
	var ymax := float(_loader.domain.get("ymax", 0.0))
	var ymin := float(_loader.domain.get("ymin", 0.0))
	var use_material := color_by == "material_id" and _mat_col >= 0
	var span := maxf(_color_hi - _color_lo, 1e-6)

	for p in _loader.particle_count:
		var base := p * stride
		# Flip Y (physics is y-up; Godot 2D is y-down) by mirroring inside the
		# domain so the camera framing below stays simple.
		var wy := ymin + ymax - data[base + _pos_y]
		var xform := Transform2D(0.0, Vector2(data[base + _pos_x], wy))
		multimesh.set_instance_transform_2d(p, xform)

		var col: Color
		if use_material:
			var mid := int(round(data[base + _mat_col]))
			col = MATERIAL_COLORS.get(mid, FALLBACK_COLOR)
		else:
			var t := clampf((data[base + _color_col] - _color_lo) / span, 0.0, 1.0)
			col = _ramp(t)
		if _damage_col >= 0:
			var dmg := clampf(data[base + _damage_col], 0.0, 1.0)
			if dmg > 0.5:
				col = col.lerp(SPALL_COLOR, (dmg - 0.5) * 2.0)
		multimesh.set_instance_color(p, col)

	_update_hud()


## Blue -> cyan -> yellow -> red perceptual-ish ramp for scalar attributes.
func _ramp(t: float) -> Color:
	if t < 0.5:
		return Color(0.1, 0.2 + 1.4 * t, 1.0 - 0.4 * t)
	return Color(0.1 + 1.8 * (t - 0.5), 0.9 - 0.9 * (t - 0.5), 0.8 - 1.6 * (t - 0.5))


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


func _compute_color_range() -> void:
	# Scalar coloring needs a normalization window; scan a mid frame for a
	# representative max (frame 0 is often at rest). Material coloring skips this.
	if color_by == "material_id" or _color_col < 0:
		return
	var f := _loader.frame_count / 2
	var data := _loader.read_frame(f)
	var stride := _loader.attributes.size()
	var hi := 0.0
	for p in _loader.particle_count:
		hi = maxf(hi, data[p * stride + _color_col])
	_color_lo = 0.0
	_color_hi = maxf(hi, 1e-6)


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
	var z := minf(vp.x / dom_w, vp.y / dom_h) * fit_margin
	_camera.zoom = Vector2(z, z)
	_camera.position = Vector2((xmin + xmax) * 0.5, (ymin + ymax) * 0.5)
	_camera.make_current()


func _setup_hud() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)
	_hud = Label.new()
	_hud.position = Vector2(12, 8)
	_hud.add_theme_color_override("font_color", Color.WHITE)
	_hud.add_theme_color_override("font_outline_color", Color.BLACK)
	_hud.add_theme_constant_override("outline_size", 4)
	layer.add_child(_hud)
	_update_hud()


func _update_hud() -> void:
	if _hud == null:
		return
	var t_ms := _frame * _loader.frame_dt * 1000.0
	_hud.text = "%s  frame %d/%d   t=%.3f ms   color:%s   %.0f fps%s" % [
		cache_dir.get_file(), _frame, _loader.frame_count - 1,
		t_ms, color_by, frames_per_second,
		"" if _playing else "   [PAUSED]",
	]


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
		_show_frame(f)
		await RenderingServer.frame_post_draw
		await RenderingServer.frame_post_draw
		var img := get_viewport().get_texture().get_image()
		var path := dir.path_join("frame_%03d.png" % f)
		var err := img.save_png(path)
		print("SHOT frame %d -> %s (%d)" % [f, path, err])
	get_tree().quit(0)
