extends MultiMeshInstance2D
## Renders the particle cloud for one cache, one draw call for the whole set
## (MultiMeshInstance2D). Colors each particle by a chosen attribute via
## per-instance color. Knows only the cache format (CLAUDE.md §2).
##
## STATUS: playback skeleton. Wire `cache_dir` to a real cache (or the golden
## fixture) and it will lay out particles and animate frames. The colormap is a
## simple built-in ramp; a shader-based colormap is a later polish step.

## Directory of the cache to play. Defaults to the golden fixture so the viewer
## runs with no solver present.
@export_dir var cache_dir: String = "res://fixtures/tiny_golden_cache"

## Attribute to color by. The loader resolves this to a column at runtime.
@export var color_by: String = "vel_mag"

## Seconds of wall-clock per render frame during playback.
@export var seconds_per_frame: float = 0.05

var _loader := CacheLoader.new()
var _frame: int = 0
var _accum: float = 0.0
var _pos_x: int = 0
var _pos_y: int = 0
var _color_col: int = -1


func _ready() -> void:
	var err = _loader.load_cache(cache_dir)
	if err != OK:
		push_error("cache load failed: %s" % str(err))
		set_process(false)
		return

	_pos_x = _loader.attribute_index("pos_x")
	_pos_y = _loader.attribute_index("pos_y")
	_color_col = _loader.attribute_index(color_by)

	multimesh = MultiMesh.new()
	multimesh.transform_format = MultiMesh.TRANSFORM_2D
	multimesh.use_colors = true
	multimesh.mesh = _make_point_mesh(1.0)
	multimesh.instance_count = _loader.particle_count

	_show_frame(0)


func _process(delta: float) -> void:
	_accum += delta
	if _accum < seconds_per_frame:
		return
	_accum = 0.0
	_frame = (_frame + 1) % _loader.frame_count
	_show_frame(_frame)


func _show_frame(f: int) -> void:
	var data := _loader.read_frame(f)
	var stride := _loader.attributes.size()
	for p in _loader.particle_count:
		var base := p * stride
		var xform := Transform2D(0.0, Vector2(data[base + _pos_x], data[base + _pos_y]))
		multimesh.set_instance_transform_2d(p, xform)
		if _color_col >= 0:
			var v := data[base + _color_col]
			multimesh.set_instance_color(p, _ramp(v))


## Placeholder color ramp. Real colormap normalization (per-attribute min/max
## from the manifest or a scan) is a later step; this just keeps values visible.
func _ramp(v: float) -> Color:
	var t := clampf(v / 2000.0, 0.0, 1.0)
	return Color(t, 0.2 + 0.5 * (1.0 - t), 1.0 - t)


func _make_point_mesh(size: float) -> QuadMesh:
	var q := QuadMesh.new()
	q.size = Vector2(size, size)
	return q
