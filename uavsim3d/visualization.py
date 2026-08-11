"""Real-time 3D visualization using pyglet with software-projected geometry."""

from __future__ import annotations

import math
from typing import List, Sequence

from .geometry import Vec3, add, cross, dot, normalize, scale, sub
from .obstacles import ExtrudedPolygonObstacle, MovingSphereObstacle3D
from .simulation import Simulation3D


def _shade(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(channel * factor))) for channel in color)


class Pyglet3DVisualizer:
    def __init__(self, simulation: Simulation3D, width: int = 1280, height: int = 820) -> None:
        try:
            import pyglet
            from pyglet import shapes
            from pyglet.window import key, mouse
        except ImportError as exc:
            raise RuntimeError("pyglet is required for the real-time 3D renderer. Install it with `./.venv/bin/pip install pyglet`.") from exc

        self.pyglet = pyglet
        self.shapes = shapes
        self.key = key
        self.mouse = mouse
        self.simulation = simulation
        self.width = width
        self.height = height
        self.aspect = width / max(1, height)
        self.fov_deg = 58.0
        self.drawables: List[object] = []
        self.overlay_drawables: List[object] = []
        self.overlay_labels: List[object] = []
        self._default_target = add(scale(self.simulation.uav.position, 0.65), scale(self.simulation.goal, 0.35))
        self.camera_target = self._default_target
        self.camera_yaw = -0.8
        self.camera_pitch = 0.42
        self.camera_radius = 88.0
        self.view_mode = "free"
        self.window = pyglet.window.Window(width=width, height=height, caption="UAV 3D Dynamic Planner", resizable=False)
        self.hud_title = self._make_label(
            "UAV Urban Navigation",
            x=26,
            y=height - 26,
            anchor_x="left",
            anchor_y="top",
            font_name="Helvetica Neue",
            font_size=18,
            bold=True,
            color=(246, 248, 252, 255),
        )
        self.hud = self._make_label(
            "",
            x=26,
            y=height - 58,
            anchor_x="left",
            anchor_y="top",
            multiline=True,
            width=330,
            font_name="Helvetica Neue",
            font_size=12,
            color=(226, 232, 240, 255),
        )
        self.legend = self._make_label(
            "",
            x=width - 24,
            y=height - 26,
            anchor_x="right",
            anchor_y="top",
            multiline=True,
            width=270,
            font_name="Helvetica Neue",
            font_size=11,
            color=(222, 230, 238, 255),
        )
        self.footer = self._make_label(
            "",
            x=26,
            y=26,
            anchor_x="left",
            anchor_y="bottom",
            font_name="Helvetica Neue",
            font_size=11,
            color=(197, 205, 216, 255),
        )
        try:
            from pyglet import gl

            gl.glClearColor(0.05, 0.07, 0.10, 1.0)
        except Exception:
            pass

        @self.window.event
        def on_draw() -> None:
            self.window.clear()
            self._draw_background()
            self._rebuild_scene()
            for drawable in self.drawables:
                drawable.draw()
            for drawable in self.overlay_drawables:
                drawable.draw()
            self.hud_title.draw()
            self.hud.draw()
            self.legend.draw()
            self.footer.draw()
            for label in self.overlay_labels:
                label.draw()

        @self.window.event
        def on_close() -> None:
            self.pyglet.clock.unschedule(self._tick)

        @self.window.event
        def on_mouse_drag(_x: int, _y: int, dx: int, dy: int, buttons: int, _modifiers: int) -> None:
            if self.view_mode != "free":
                self._detach_follow_view()
            if buttons & self.mouse.LEFT:
                self.camera_yaw -= dx * 0.008
                self.camera_pitch = self._clamp_pitch(self.camera_pitch + dy * 0.006)
            elif buttons & self.mouse.RIGHT or buttons & self.mouse.MIDDLE:
                self._pan_camera(-dx * 0.08, -dy * 0.08)

        @self.window.event
        def on_mouse_scroll(_x: int, _y: int, _scroll_x: float, scroll_y: float) -> None:
            if self.view_mode != "free":
                self._detach_follow_view()
            zoom_scale = 0.92 if scroll_y > 0 else 1.08
            self.camera_radius = max(18.0, min(220.0, self.camera_radius * zoom_scale))

        @self.window.event
        def on_key_press(symbol: int, _modifiers: int) -> None:
            if symbol == self.key._0:
                self.view_mode = "free"
            elif symbol == self.key._1:
                self.view_mode = "follow_back"
            elif symbol == self.key._2:
                self.view_mode = "follow_front"
            elif symbol == self.key._3:
                self.view_mode = "follow_left"
            elif symbol == self.key._4:
                self.view_mode = "follow_right"
            elif symbol == self.key._5:
                self.view_mode = "follow_top"
            elif symbol == self.key.LEFT:
                if self.view_mode != "free":
                    self._detach_follow_view()
                self.camera_yaw += 0.12
            elif symbol == self.key.RIGHT:
                if self.view_mode != "free":
                    self._detach_follow_view()
                self.camera_yaw -= 0.12
            elif symbol == self.key.UP:
                if self.view_mode != "free":
                    self._detach_follow_view()
                self.camera_pitch = self._clamp_pitch(self.camera_pitch + 0.10)
            elif symbol == self.key.DOWN:
                if self.view_mode != "free":
                    self._detach_follow_view()
                self.camera_pitch = self._clamp_pitch(self.camera_pitch - 0.10)
            elif symbol in {self.key.EQUAL, self.key.NUM_ADD}:
                if self.view_mode != "free":
                    self._detach_follow_view()
                self.camera_radius = max(18.0, self.camera_radius * 0.9)
            elif symbol in {self.key.MINUS, self.key.NUM_SUBTRACT}:
                if self.view_mode != "free":
                    self._detach_follow_view()
                self.camera_radius = min(220.0, self.camera_radius * 1.1)
            elif symbol == self.key.R:
                self._reset_camera()

        self.pyglet.clock.schedule_interval(self._tick, self.simulation.config.dt)

    def run(self) -> None:
        self.pyglet.app.run()

    def _tick(self, _dt: float) -> None:
        if not self.simulation.finished:
            self.simulation.step()

    def _rebuild_scene(self) -> None:
        self.drawables.clear()
        self.overlay_drawables.clear()
        self.overlay_labels.clear()
        camera, target = self._camera()
        forward, right, up = self._camera_basis(camera, target)
        items: List[tuple[float, object]] = []

        items.extend(self._ground_drawables(camera, forward, right, up))
        for obstacle in self.simulation.env.static_obstacles:
            items.extend(self._building_drawables(obstacle, camera, forward, right, up))
        for obstacle in self.simulation.env.dynamic_spheres:
            items.extend(self._sphere_drawables(obstacle, camera, forward, right, up))
        for flock in self.simulation.env.flocks:
            for obstacle in flock.obstacle_obstacles():
                items.extend(self._sphere_drawables(obstacle, camera, forward, right, up))

        items.extend(self._beacon_drawables(self.simulation.metrics.trajectory[0], "START", camera, forward, right, up, (91, 180, 255)))
        items.extend(self._beacon_drawables(self.simulation.goal, "GOAL", camera, forward, right, up, (250, 204, 21)))
        items.extend(self._polyline_drawables(self.simulation.metrics.trajectory, camera, forward, right, up, (102, 163, 255), width=3))
        if self.simulation.path:
            items.extend(self._polyline_drawables(self.simulation.path, camera, forward, right, up, (66, 215, 170), width=2))
        items.extend(self._uav_drawables(camera, forward, right, up))

        self.drawables = [drawable for _, drawable in sorted(items, key=lambda item: item[0], reverse=True)]
        clearance = "clear" if math.isinf(self.simulation.nearest_distance) else f"{self.simulation.nearest_distance:.2f} m"
        hud_lines = [
            "Planner: RRT* + PSO",
            f"Time: {self.simulation.sim_time:5.1f} s",
            f"Speed: {self.simulation.current_speed:4.2f} m/s",
            f"Clearance: {clearance}",
            f"Replans: {self.simulation.metrics.replans}",
            f"View: {self._view_label()}",
        ]
        if self.simulation.finished:
            hud_lines.append("State: collision" if self.simulation.collided else "State: goal reached")
        self.hud.text = "\n".join(hud_lines)
        self.legend.text = "\n".join(
            [
                "LEGEND",
                "Blue path: flown trajectory",
                "Mint path: active plan",
                "Amber beacon: destination",
                "Orange / red spheres: dynamic obstacles",
                "1-5: cinematic follow views",
                "0: free camera   R: reset",
            ]
        )
        path_nodes = 0 if not self.simulation.path else len(self.simulation.path)
        self.footer.text = (
            f"Urban canyon scene   |   waypoint nodes: {path_nodes}   |   "
            f"step: {self.simulation.step_count}   |   start to goal mission"
        )
        self._rebuild_overlay_panels()

    def _camera(self) -> tuple[Vec3, Vec3]:
        if self.view_mode != "free":
            return self._follow_camera()
        target = self.camera_target
        horizontal = self.camera_radius * math.cos(self.camera_pitch)
        camera = (
            target[0] + horizontal * math.cos(self.camera_yaw),
            target[1] + horizontal * math.sin(self.camera_yaw),
            target[2] + self.camera_radius * math.sin(self.camera_pitch),
        )
        return camera, target

    def _follow_camera(self) -> tuple[Vec3, Vec3]:
        heading = self._uav_heading()
        planar_heading = normalize((heading[0], heading[1], 0.0))
        if math.isclose(dot(planar_heading, planar_heading), 0.0):
            planar_heading = (1.0, 0.0, 0.0)
        world_up = (0.0, 0.0, 1.0)
        right = normalize(cross(planar_heading, world_up))
        if math.isclose(dot(right, right), 0.0):
            right = (0.0, 1.0, 0.0)
        target = add(self.simulation.uav.position, (0.0, 0.0, 2.0))
        look_ahead = add(target, scale(planar_heading, 6.0))
        if self.view_mode == "follow_back":
            return add(target, add(scale(planar_heading, -18.0), (0.0, 0.0, 7.0))), look_ahead
        if self.view_mode == "follow_front":
            return add(target, add(scale(planar_heading, 16.0), (0.0, 0.0, 5.5))), target
        if self.view_mode == "follow_left":
            return add(target, add(scale(right, -16.0), (0.0, 0.0, 6.0))), look_ahead
        if self.view_mode == "follow_right":
            return add(target, add(scale(right, 16.0), (0.0, 0.0, 6.0))), look_ahead
        return add(target, (0.0, 0.0, 24.0)), target

    @staticmethod
    def _camera_basis(camera: Vec3, target: Vec3) -> tuple[Vec3, Vec3, Vec3]:
        forward = normalize(sub(target, camera))
        world_up = (0.0, 0.0, 1.0)
        right = normalize(cross(forward, world_up))
        if math.isclose(dot(right, right), 0.0):
            right = (1.0, 0.0, 0.0)
        up = normalize(cross(right, forward))
        return forward, right, up

    def _project(self, point: Vec3, camera: Vec3, forward: Vec3, right: Vec3, up: Vec3) -> tuple[float, float, float] | None:
        rel = sub(point, camera)
        x_cam = dot(rel, right)
        y_cam = dot(rel, up)
        z_cam = dot(rel, forward)
        if z_cam <= 0.5:
            return None
        f = 1.0 / math.tan(math.radians(self.fov_deg) / 2.0)
        x_ndc = (x_cam * f / self.aspect) / z_cam
        y_ndc = (y_cam * f) / z_cam
        sx = (x_ndc + 1.0) * 0.5 * self.width
        sy = (y_ndc + 1.0) * 0.5 * self.height
        return (sx, sy, z_cam)

    def _building_drawables(
        self,
        obstacle: ExtrudedPolygonObstacle,
        camera: Vec3,
        forward: Vec3,
        right: Vec3,
        up: Vec3,
    ) -> List[tuple[float, object]]:
        light = normalize((0.35, 0.40, 0.85))
        items: List[tuple[float, object]] = []
        for a, b, c in obstacle.surface_triangles():
            pa = self._project(a, camera, forward, right, up)
            pb = self._project(b, camera, forward, right, up)
            pc = self._project(c, camera, forward, right, up)
            if pa is None or pb is None or pc is None:
                continue
            normal = normalize(cross(sub(b, a), sub(c, a)))
            shade = 0.45 + 0.5 * max(0.0, dot(normal, light))
            color = _shade(obstacle.color, shade)
            tri = self.shapes.Triangle(pa[0], pa[1], pb[0], pb[1], pc[0], pc[1], color=color)
            avg_depth = (pa[2] + pb[2] + pc[2]) / 3.0
            items.append((avg_depth, tri))
        top_ring = obstacle.footprint
        for a_xy, b_xy in zip(top_ring, top_ring[1:] + top_ring[:1]):
            a_top = self._project((a_xy[0], a_xy[1], obstacle.z_max), camera, forward, right, up)
            b_top = self._project((b_xy[0], b_xy[1], obstacle.z_max), camera, forward, right, up)
            if a_top is None or b_top is None:
                continue
            edge = self._make_line(a_top[0], a_top[1], b_top[0], b_top[1], 1, _shade(obstacle.color, 1.4))
            edge.opacity = 150
            items.append((((a_top[2] + b_top[2]) / 2.0) - 0.02, edge))
        return items

    def _sphere_drawables(
        self,
        obstacle: MovingSphereObstacle3D,
        camera: Vec3,
        forward: Vec3,
        right: Vec3,
        up: Vec3,
    ) -> List[tuple[float, object]]:
        items: List[tuple[float, object]] = []
        if len(obstacle.trail) > 1:
            items.extend(self._polyline_drawables(obstacle.trail, camera, forward, right, up, obstacle.color, width=1))
        projected = self._project(obstacle.center, camera, forward, right, up)
        if projected is None:
            return items
        pixel_radius = max(2.0, (260.0 * obstacle.radius) / projected[2])
        glow = self.shapes.Circle(projected[0], projected[1], pixel_radius * 1.55, color=_shade(obstacle.color, 0.55))
        glow.opacity = 80
        circle = self.shapes.Circle(projected[0], projected[1], pixel_radius, color=obstacle.color)
        items.append((projected[2], glow))
        items.append((projected[2] - 0.01, circle))
        return items

    def _uav_drawables(self, camera: Vec3, forward: Vec3, right: Vec3, up: Vec3) -> List[tuple[float, object]]:
        projected = self._project(self.simulation.uav.position, camera, forward, right, up)
        if projected is None:
            return []
        color = (98, 247, 205) if not self.simulation.finished else (250, 204, 21)
        shadow_point = self._project((self.simulation.uav.position[0], self.simulation.uav.position[1], 0.0), camera, forward, right, up)
        heading = self._uav_heading()
        nose_world = add(self.simulation.uav.position, scale(heading, 2.6))
        nose = self._project(nose_world, camera, forward, right, up)
        items: List[tuple[float, object]] = []
        if shadow_point is not None:
            shadow = self.shapes.Circle(shadow_point[0], shadow_point[1], 10.0, color=(12, 18, 24))
            shadow.opacity = 90
            items.append((shadow_point[2] + 0.05, shadow))
        outer = self.shapes.Circle(projected[0], projected[1], 11.0, color=(41, 121, 255))
        outer.opacity = 120
        halo = self.shapes.Circle(projected[0], projected[1], 7.2, color=(26, 188, 156))
        halo.opacity = 170
        core = self.shapes.Circle(projected[0], projected[1], 4.4, color=color)
        items.extend([(projected[2], outer), (projected[2] - 0.01, halo), (projected[2] - 0.02, core)])
        if nose is not None:
            heading_line = self._make_line(projected[0], projected[1], nose[0], nose[1], 2, (244, 247, 250))
            heading_line.opacity = 180
            items.append((((projected[2] + nose[2]) / 2.0) - 0.03, heading_line))
        return items

    def _polyline_drawables(
        self,
        points: Sequence[Vec3],
        camera: Vec3,
        forward: Vec3,
        right: Vec3,
        up: Vec3,
        color: tuple[int, int, int],
        width: int = 1,
    ) -> List[tuple[float, object]]:
        items: List[tuple[float, object]] = []
        for a, b in zip(points, points[1:]):
            pa = self._project(a, camera, forward, right, up)
            pb = self._project(b, camera, forward, right, up)
            if pa is None or pb is None:
                continue
            line = self._make_line(pa[0], pa[1], pb[0], pb[1], width, color)
            items.append((((pa[2] + pb[2]) / 2.0), line))
        return items

    def _make_line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        width: int,
        color: tuple[int, int, int],
    ) -> object:
        try:
            return self.shapes.Line(x1, y1, x2, y2, thickness=width, color=color)
        except TypeError:
            return self.shapes.Line(x1, y1, x2, y2, width=width, color=color)

    def _ground_drawables(
        self,
        camera: Vec3,
        forward: Vec3,
        right: Vec3,
        up: Vec3,
    ) -> List[tuple[float, object]]:
        items: List[tuple[float, object]] = []
        env = self.simulation.env
        for x in range(int(env.x_min), int(env.x_max) + 1, 10):
            start = self._project((float(x), env.y_min, 0.0), camera, forward, right, up)
            end = self._project((float(x), env.y_max, 0.0), camera, forward, right, up)
            if start is None or end is None:
                continue
            major = x % 20 == 0
            color = (48, 88, 108) if major else (34, 60, 76)
            width = 2 if major else 1
            line = self._make_line(start[0], start[1], end[0], end[1], width, color)
            line.opacity = 120 if major else 70
            items.append((((start[2] + end[2]) / 2.0) + 0.3, line))
        for y in range(int(env.y_min), int(env.y_max) + 1, 10):
            start = self._project((env.x_min, float(y), 0.0), camera, forward, right, up)
            end = self._project((env.x_max, float(y), 0.0), camera, forward, right, up)
            if start is None or end is None:
                continue
            major = y % 20 == 0
            color = (48, 88, 108) if major else (34, 60, 76)
            width = 2 if major else 1
            line = self._make_line(start[0], start[1], end[0], end[1], width, color)
            line.opacity = 120 if major else 70
            items.append((((start[2] + end[2]) / 2.0) + 0.3, line))
        return items

    def _beacon_drawables(
        self,
        point: Vec3,
        label_text: str,
        camera: Vec3,
        forward: Vec3,
        right: Vec3,
        up: Vec3,
        color: tuple[int, int, int],
    ) -> List[tuple[float, object]]:
        items: List[tuple[float, object]] = []
        top_point = (point[0], point[1], point[2] + 9.0)
        ground = self._project(point, camera, forward, right, up)
        tip = self._project(top_point, camera, forward, right, up)
        if ground is None or tip is None:
            return items
        stem = self._make_line(ground[0], ground[1], tip[0], tip[1], 2, color)
        stem.opacity = 170
        ring = self.shapes.Circle(ground[0], ground[1], 7.0, color=color)
        ring.opacity = 120
        dot = self.shapes.Circle(ground[0], ground[1], 3.2, color=(245, 248, 250))
        label = self._make_label(
            label_text,
            x=tip[0],
            y=tip[1] + 10,
            anchor_x="center",
            anchor_y="bottom",
            font_name="Helvetica Neue",
            font_size=10,
            bold=True,
            color=(245, 248, 250, 230),
        )
        items.extend(
            [
                (((ground[2] + tip[2]) / 2.0) - 0.02, stem),
                (ground[2] - 0.03, ring),
                (ground[2] - 0.04, dot),
                (tip[2] - 0.05, label),
            ]
        )
        return items

    def _draw_background(self) -> None:
        top = self.shapes.Rectangle(0, self.height * 0.45, self.width, self.height * 0.55, color=(7, 12, 20))
        top.opacity = 255
        mid = self.shapes.Rectangle(0, self.height * 0.22, self.width, self.height * 0.23, color=(12, 26, 38))
        mid.opacity = 220
        low = self.shapes.Rectangle(0, 0, self.width, self.height * 0.22, color=(10, 18, 24))
        low.opacity = 255
        glow = self.shapes.Circle(self.width * 0.82, self.height * 0.72, 120.0, color=(24, 52, 72))
        glow.opacity = 55
        top.draw()
        mid.draw()
        low.draw()
        glow.draw()

    def _rebuild_overlay_panels(self) -> None:
        left_panel = self.shapes.Rectangle(14, self.height - 178, 352, 152, color=(11, 18, 27))
        left_panel.opacity = 188
        right_panel = self.shapes.Rectangle(self.width - 308, self.height - 158, 294, 132, color=(11, 18, 27))
        right_panel.opacity = 176
        footer_panel = self.shapes.Rectangle(14, 12, 488, 32, color=(11, 18, 27))
        footer_panel.opacity = 160
        self.overlay_drawables.extend([left_panel, right_panel, footer_panel])

    def _make_label(self, text: str, **kwargs: object) -> object:
        try:
            return self.pyglet.text.Label(text, **kwargs)
        except TypeError:
            fallback = dict(kwargs)
            fallback.pop("bold", None)
            return self.pyglet.text.Label(text, **fallback)

    def _uav_heading(self) -> Vec3:
        trajectory = self.simulation.metrics.trajectory
        if len(trajectory) >= 2:
            delta = sub(trajectory[-1], trajectory[-2])
            if dot(delta, delta) > 1e-6:
                return normalize(delta)
        if self.simulation.path and len(self.simulation.path) > 1:
            return normalize(sub(self.simulation.path[1], self.simulation.uav.position))
        fallback = sub(self.simulation.goal, self.simulation.uav.position)
        if dot(fallback, fallback) <= 1e-6:
            return (1.0, 0.0, 0.0)
        return normalize(fallback)

    def _detach_follow_view(self) -> None:
        camera, target = self._camera()
        offset = sub(camera, target)
        horizontal = math.hypot(offset[0], offset[1])
        self.camera_radius = max(18.0, min(220.0, math.sqrt(dot(offset, offset))))
        self.camera_yaw = math.atan2(offset[1], offset[0])
        self.camera_pitch = self._clamp_pitch(math.atan2(offset[2], max(1e-6, horizontal)))
        self.camera_target = target
        self.view_mode = "free"

    def _pan_camera(self, dx: float, dy: float) -> None:
        camera, target = self._camera()
        _forward, right, up = self._camera_basis(camera, target)
        self.camera_target = add(self.camera_target, add(scale(right, dx), scale(up, dy)))

    def _reset_camera(self) -> None:
        self.view_mode = "free"
        self.camera_target = self._default_target
        self.camera_yaw = -0.8
        self.camera_pitch = 0.42
        self.camera_radius = 88.0

    def _view_label(self) -> str:
        labels = {
            "free": "free",
            "follow_back": "back",
            "follow_front": "front",
            "follow_left": "left",
            "follow_right": "right",
            "follow_top": "top",
        }
        return labels.get(self.view_mode, self.view_mode)

    @staticmethod
    def _clamp_pitch(pitch: float) -> float:
        return max(-1.2, min(1.2, pitch))
