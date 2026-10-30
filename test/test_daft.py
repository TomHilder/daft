import daft
import numpy as np
import pytest


def test_auto_size():
    with daft.PGM() as pgm:
        pgm.add_node("node1", x=0.0, y=0.0)
        pgm.add_node("node2", x=1.0, y=0.0)
        pgm.add_edge("node1", "node2")
        pgm.render()


def test_manual_size():
    with daft.PGM(shape=[1, 1], origin=[0, 0]) as pgm:
        pgm.add_node("node1", x=0.0, y=0.0)
        pgm.add_node("node2", x=1.0, y=0.0)
        pgm.add_edge("node1", "node2")
        pgm.render()


def test_add_node():
    with daft.PGM() as pgm:
        pgm.add_node(node="node1", content="content1")
        assert "node1" in pgm._nodes
        assert pgm._nodes["node1"].content == "content1"


def test_add_node_class():
    with daft.PGM() as pgm:
        node = daft.Node(name="node2", content="node2", x=0, y=0)
        pgm.add_node(node)
        assert "node2" in pgm._nodes


def test_overlap_nodes():
    with daft.PGM() as pgm:
        pgm.add_node("node1", x=0, y=0)
        pgm.add_node("node2", x=0, y=0)
        pgm.add_edge("node1", "node2")
        with pytest.raises(daft.SameLocationError):
            pgm.render()


def test_add_edge():
    with daft.PGM() as pgm:
        pgm.add_node("node1")
        pgm.add_node("node2")
        pgm.add_edge(name1="node1", name2="node2")

        edge = pgm._edges[0]
        assert edge.node1 == pgm._nodes["node1"]
        assert edge.node2 == pgm._nodes["node2"]


def test_add_plate():
    with daft.PGM() as pgm:
        pgm.add_plate([0, 0, 1, 1])
        assert pgm._plates[0].rect == [0, 0, 1, 1]


def test_add_text():
    with daft.PGM() as pgm:
        pgm.add_text(x=0, y=0, label="text1")

        plate = pgm._plates[0]
        assert plate.rect == [0, 0, 0, 0]
        assert plate.label == "text1"


def test_pop_multiple():
    _dict = {"ec": "none", "edgecolor": "none"}
    with pytest.raises(TypeError):
        daft._pop_multiple(_dict, "none", "ec", "edgecolor")


def _arrow_end_offsets(angle, line_width, plot_params, shape, aspect):
    """
    Render a single arrow at 508 dpi (2 pixels per 0.01 cm) with invisible
    nodes, and return the distances, in pixels, by which its ink reaches past
    the outline of the node it starts from and of the node it points at,
    measured along the arrow. Negative means it stops short.
    """
    dpi = 508
    theta = np.deg2rad(angle)
    u = np.array([np.cos(theta), np.sin(theta)])
    # The integer shape and origin make the figure a whole number of pixels.
    pgm = daft.PGM(
        shape=[4, 4], origin=[-2, -2], line_width=line_width, dpi=dpi
    )
    invisible = {"ec": "none", "fc": "none"}
    for name, sign in [("a", -1), ("b", 1)]:
        pgm.add_node(
            name,
            x=sign * 0.75 * u[0],
            y=sign * 0.75 * u[1],
            shape=shape,
            aspect=aspect,
            plot_params=invisible,
        )
    pgm.add_edge("a", "b", plot_params=plot_params)
    ax = pgm.render()
    pgm.figure.patch.set_facecolor("white")
    pgm.figure.canvas.draw()
    ink = 1 - np.asarray(pgm.figure.canvas.buffer_rgba())[..., 0] / 255

    # Where the centre line crosses the node outlines, in plot coordinates.
    half_height = 0.5 * pgm._ctx.node_unit
    half_width = aspect * half_height
    if shape == "ellipse":
        dist = 1 / np.hypot(u[0] / half_width, u[1] / half_height)
    else:
        with np.errstate(divide="ignore"):
            dist = min(half_width / abs(u[0]), half_height / abs(u[1]))
    tail = pgm._ctx.convert(-0.75 * u[0], -0.75 * u[1]) + dist * u
    tip = pgm._ctx.convert(0.75 * u[0], 0.75 * u[1]) - dist * u
    tail, tip = ax.transData.transform([tail, tip])

    # Centres of pixels with ink, in display coordinates. A low threshold
    # finds the sharp tip, which is narrower than a pixel near its point.
    rows, cols = np.nonzero(ink >= 0.1)
    xy = np.stack([cols + 0.5, ink.shape[0] - rows - 0.5], axis=1)
    along = xy[:, 0] * u[0] + xy[:, 1] * u[1]
    return np.max(tail @ u - along), np.max(along - tip @ u)


@pytest.mark.parametrize("angle", [0, 30, 90, 160, 225, 300])
@pytest.mark.parametrize("line_width", [0.5, 1.0, 3.0])
@pytest.mark.parametrize(
    "plot_params",
    [
        {},
        {"head_length": 0.1, "head_width": 0.15},
        {"head_length": 0.4, "head_width": 0.08},
        {"overhang": 0.3},
        {"linestyle": "--"},
        {"fc": "none"},
    ],
)
@pytest.mark.parametrize(
    "shape, aspect", [("ellipse", 1.0), ("ellipse", 1.7), ("rectangle", 1.3)]
)
def test_arrow_ends_on_node_outlines(
    angle, line_width, plot_params, shape, aspect
):
    tail, tip = _arrow_end_offsets(
        angle, line_width, plot_params, shape, aspect
    )
    assert abs(tip) < 2
    assert abs(tail) < 2
