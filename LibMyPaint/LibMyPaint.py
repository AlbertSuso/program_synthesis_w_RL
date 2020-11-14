import os
import collections
import enum
import copy
from LibMyPaint import utils
import numpy as np

from dm_env import specs
from spiral.environments import pylibmypaint


class BrushSettings(enum.IntEnum):
    """Enumeration of brush settings."""

    (MYPAINT_BRUSH_SETTING_OPAQUE,
     MYPAINT_BRUSH_SETTING_OPAQUE_MULTIPLY,
     MYPAINT_BRUSH_SETTING_OPAQUE_LINEARIZE,
     MYPAINT_BRUSH_SETTING_RADIUS_LOGARITHMIC,
     MYPAINT_BRUSH_SETTING_HARDNESS,
     MYPAINT_BRUSH_SETTING_ANTI_ALIASING,
     MYPAINT_BRUSH_SETTING_DABS_PER_BASIC_RADIUS,
     MYPAINT_BRUSH_SETTING_DABS_PER_ACTUAL_RADIUS,
     MYPAINT_BRUSH_SETTING_DABS_PER_SECOND,
     MYPAINT_BRUSH_SETTING_RADIUS_BY_RANDOM,
     MYPAINT_BRUSH_SETTING_SPEED1_SLOWNESS,
     MYPAINT_BRUSH_SETTING_SPEED2_SLOWNESS,
     MYPAINT_BRUSH_SETTING_SPEED1_GAMMA,
     MYPAINT_BRUSH_SETTING_SPEED2_GAMMA,
     MYPAINT_BRUSH_SETTING_OFFSET_BY_RANDOM,
     MYPAINT_BRUSH_SETTING_OFFSET_BY_SPEED,
     MYPAINT_BRUSH_SETTING_OFFSET_BY_SPEED_SLOWNESS,
     MYPAINT_BRUSH_SETTING_SLOW_TRACKING,
     MYPAINT_BRUSH_SETTING_SLOW_TRACKING_PER_DAB,
     MYPAINT_BRUSH_SETTING_TRACKING_NOISE,
     MYPAINT_BRUSH_SETTING_COLOR_H,
     MYPAINT_BRUSH_SETTING_COLOR_S,
     MYPAINT_BRUSH_SETTING_COLOR_V,
     MYPAINT_BRUSH_SETTING_RESTORE_COLOR,
     MYPAINT_BRUSH_SETTING_CHANGE_COLOR_H,
     MYPAINT_BRUSH_SETTING_CHANGE_COLOR_L,
     MYPAINT_BRUSH_SETTING_CHANGE_COLOR_HSL_S,
     MYPAINT_BRUSH_SETTING_CHANGE_COLOR_V,
     MYPAINT_BRUSH_SETTING_CHANGE_COLOR_HSV_S,
     MYPAINT_BRUSH_SETTING_SMUDGE,
     MYPAINT_BRUSH_SETTING_SMUDGE_LENGTH,
     MYPAINT_BRUSH_SETTING_SMUDGE_RADIUS_LOG,
     MYPAINT_BRUSH_SETTING_ERASER,
     MYPAINT_BRUSH_SETTING_STROKE_THRESHOLD,
     MYPAINT_BRUSH_SETTING_STROKE_DURATION_LOGARITHMIC,
     MYPAINT_BRUSH_SETTING_STROKE_HOLDTIME,
     MYPAINT_BRUSH_SETTING_CUSTOM_INPUT,
     MYPAINT_BRUSH_SETTING_CUSTOM_INPUT_SLOWNESS,
     MYPAINT_BRUSH_SETTING_ELLIPTICAL_DAB_RATIO,
     MYPAINT_BRUSH_SETTING_ELLIPTICAL_DAB_ANGLE,
     MYPAINT_BRUSH_SETTING_DIRECTION_FILTER,
     MYPAINT_BRUSH_SETTING_LOCK_ALPHA,
     MYPAINT_BRUSH_SETTING_COLORIZE,
     MYPAINT_BRUSH_SETTING_SNAP_TO_PIXEL,
     MYPAINT_BRUSH_SETTING_PRESSURE_GAIN_LOG,
     MYPAINT_BRUSH_SETTINGS_COUNT) = range(46)


def _fix15_to_rgba(buf):
    """Converts buffer from a 15-bit fixed-point representation into uint8 RGBA.
    Taken verbatim from the C code for libmypaint.
    Args:
      buf: 15-bit fixed-point buffer represented in `uint16`.
    Returns:
      A `uint8` buffer with RGBA channels.
    """
    rgb, alpha = np.split(buf, [3], axis=2)
    rgb = rgb.astype(np.uint32)
    mask = alpha[..., 0] == 0
    rgb[mask] = 0
    rgb[~mask] = ((rgb[~mask] << 15) + alpha[~mask] // 2) // alpha[~mask]
    rgba = np.concatenate((rgb, alpha), axis=2)
    rgba = (255 * rgba + (1 << 15) // 2) // (1 << 15)
    return rgba.astype(np.uint8)


class Environment:
    """
    Wrapper de la libreria libmypaint diseñado como environment para un agente de RL.

    Su interfície consta de tres métodos:
       - observation(self): Devuelve un orderedDict con el canvas y el numero de step actual.
       - reset(self): Resetea el environment al estado inicial y lo deja listo para usar.
       - step(self, action): realiza un step (i.e. realiza la acción action). La acción debe
                             proveerse en un formato concreto (especificado por el parámetro
                             self._action_spec)
    """

    # Numero de strokes (trazos de pincel) por cada step (por cada curva de bezier dibujada)
    STROKES_PER_STEP = 50
    DTIME = 0.1

    # Define los valores válidos de presión y RGB
    P_VALUES = np.linspace(0.1, 1.0, 10)
    R_VALUES = np.linspace(0.0, 1.0, 20)
    G_VALUES = np.linspace(0.0, 1.0, 20)
    B_VALUES = np.linspace(0.0, 1.0, 20)

    def __init__(self, canvas_width, grid_width, brush_type,
                 use_color, brush_sizes=(1, 2, 3), use_pressure=True, use_alpha=False,
                 background="white", brushes_basedir=""):

        """
        :param canvas_width:(int) tamaño del canvas (será cuadrado)
        :param grid_width:(int) tamaño del grid (será cuadrado). El grid es una abstracción del
                           canvas, para que los usuarios puedan suponer que este siempre tiene el
                           mismo tamaño. Las "coordenadas grid" se transforman a
                           "coordenadas canvas" con el método "grid_to_real(self, location)"
        :param brush_type:(str) tipo de pincel (podrá cambiarse)
        :param use_color:(bool)
        :param brush_sizes:(int list)
        :param use_pressure:(bool)
        :param use_alpha:(bool)
        :param background:(str)
        :param brushes_basedir:(path)
        """

        # Atributos correspondientes al canvas
        self._background = background
        self._canvas_width = canvas_width
        self._grid_width = grid_width
        self._use_color = use_color
        self._use_alpha = use_alpha
        if not self._use_color:
            self._output_channels = 1
        elif not self._use_alpha:
            self._output_channels = 3
        else:
            self._output_channels = 4

        # Inicializamos el canvas
        if background == "white":
            background = pylibmypaint.SurfaceWrapper.Background.kWhite
        elif background == "transparent":
            background = pylibmypaint.SurfaceWrapper.Background.kBlack
        else:
            raise ValueError("Invalid background type: {}".format(background))
        self._surface = pylibmypaint.SurfaceWrapper(self._canvas_width, self._canvas_width, background)

        # Atributos correspondientes a la brush (_use_color y _use_alpha también podrían ir aquí)
        self._use_pressure = use_pressure
        assert np.all(np.array(brush_sizes) > 0.)
        self._log_brush_sizes = np.array([np.log(float(i)) for i in brush_sizes])
        self._use_size = True if len(self._log_brush_sizes) > 1 else False

        # Parámetros asociados a la brush. De ellos, los que son parámetros del objeto
        # BrushWrapper() contenido en self._brush son x,y,hue,saturation,value,log_size.
        hue, saturation, value = utils.rgb_to_hsv(self.R_VALUES[0], self.G_VALUES[0], self.B_VALUES[0])
        pressure = 0.0 if self._use_pressure else 1
        self._brush_params = collections.OrderedDict([
            ("y", 0.0),
            ("x", 0.0),
            ("pressure", pressure),
            ("log_size", self._log_brush_sizes[0]),
            ("hue", hue),
            ("saturation", saturation),
            ("value", value),
            ("is_painting", False)])
        self._prev_brush_params = None

        # Inicializamos la brush
        self._brush = pylibmypaint.BrushWrapper()
        self._brush.SetSurface(self._surface)
        self._brush.LoadFromFile(
            os.path.join(brushes_basedir, "brushes/{}.myb".format(brush_type)))

        """
        Atributos correspondientes a las acciones y al estado del environment.
        El atributo "_action_spec" determina como debe ser el formato de las acciones 
        que podrán aplicarse sobre el environment (specs.DiscreteArray(n) determina que el valor deberá ser
        un escalar con dtype=int32 y estar entre 0 y n-1 incluidos).
        """
        self._episode_step = 0
        self.stats = {"total_strokes": 0, "total_disjoint": 0}
        self._action_spec = collections.OrderedDict([
            ("control", specs.DiscreteArray(self._grid_width*self._grid_width)),
            ("end", specs.DiscreteArray(self._grid_width*self._grid_width)),
            ("flag", specs.DiscreteArray(2)),
            ("pressure", specs.DiscreteArray(len(self.P_VALUES))),
            ("size", specs.DiscreteArray(len(self._log_brush_sizes))),
            ("red", specs.DiscreteArray(len(self.R_VALUES))),
            ("green", specs.DiscreteArray(len(self.G_VALUES))),
            ("blue", specs.DiscreteArray(len(self.B_VALUES)))])

        if not self._use_pressure:
            del self._action_spec["pressure"]
        if not self._use_size:
            del self._action_spec["size"]
        if not self._use_color:
            del self._action_spec["red"]
            del self._action_spec["green"]
            del self._action_spec["blue"]

        # Dejamos el environment listo para usar
        self.reset()

    @property
    def action_spec(self):
        return self._action_spec

    @property
    def canvas_width(self):
        return self._canvas_width

    @property
    def num_channels(self):
        return self._output_channels

    def _get_canvas(self):
        """
        :return: canvas en forma de numpy array
        """
        buf = self._surface.BufferAsNumpy()
        buf = buf.transpose((0, 2, 1, 3, 4))
        size = self._canvas_width if self._canvas_width % 64 == 0 else (self._canvas_width//64+1)*64

        buf = buf.reshape((size, size, 4))
        canvas = np.single(_fix15_to_rgba(buf)) / 255.0
        return canvas

    def observation(self):
        """
        :return: Estado actual del environment. Devuelve un orderedDict con el canvas y
                 el episode_step_num actual
        """
        canvas = self._get_canvas()
        if not self._use_color:
            canvas = canvas[..., 0:1]
        elif not self._use_alpha:
            canvas = canvas[..., 0:3]

        episode_step = np.array(self._episode_step, dtype=np.int32)
        canvas = canvas[0:self._canvas_width, 0:self._canvas_width]

        return collections.OrderedDict([
            ("canvas", canvas),
            ("episode_step", episode_step)])

    def _update_libmypaint_brush(self, **kwargs):
        """

        :param kwargs: argumentos no posicionales, entre los cuales pueden estar "log_size",
                       "hue", "saturation" y "value".
        Actualiza los valores del objeto brush almacenado en self._brush.
        """
        if "log_size" in kwargs:
            self._brush.SetBaseValue(
                BrushSettings.MYPAINT_BRUSH_SETTING_RADIUS_LOGARITHMIC,
                kwargs["log_size"])

        hsv_keys = ["hue", "saturation", "value"]
        if any(k in kwargs for k in hsv_keys):
            assert all(k in kwargs for k in hsv_keys)
            self._brush.SetBaseValue(
                BrushSettings.MYPAINT_BRUSH_SETTING_COLOR_H, kwargs["hue"])
            self._brush.SetBaseValue(
                BrushSettings.MYPAINT_BRUSH_SETTING_COLOR_S, kwargs["saturation"])
            self._brush.SetBaseValue(
                BrushSettings.MYPAINT_BRUSH_SETTING_COLOR_V, kwargs["value"])

    def _update_brush_params(self, **kwargs):
        """
        :param kwargs: argumentos no posicionales. Vale cualquiera de los parámetros de la brush,
                       es decir, vale cualquier key de self._brush_params.
        Actualiza tanto los valores de self._brush_params y self._prev_brush_params como los
        valores "log_size", "hue", "saturation" y "value" del objeto brush contenido en self._brush.
        """
        rgb_keys = ["red", "green", "blue"]

        if any(k in kwargs for k in rgb_keys):
            assert all(k in kwargs for k in rgb_keys)
            red, green, blue = [kwargs[k] for k in rgb_keys]
            for k in rgb_keys:
                del kwargs[k]
            if self._use_color:
                hue, saturation, value = utils.rgb_to_hsv(red, green, blue)
                kwargs.update(dict(hue=hue, saturation=saturation, value=value))

        # Actualizamos los valores de estado de la brush
        self._prev_brush_params = copy.copy(self._brush_params)
        self._brush_params.update(kwargs)

        if not self._prev_brush_params["is_painting"]:
            # Si antes no estabamos pintando, haremos como que la apariencia de la brush
            # no ha cambiado.
            self._prev_brush_params.update({
                k: self._brush_params[k] for k in ["pressure", "log_size",
                                                   "hue", "saturation", "value"]})

        # Actualizamos el objeto self._brush
        self._update_libmypaint_brush(**kwargs)

    def _move_to(self, y, x, update_brush_params=True):
        """
        Mueve la brush a las "coordenadas canvas" indicadas. No realiza ningún cambio en el resto
        de parámetros del objeto brush. Realiza cambios en los parámetros x,y,is_painting de
        self._brush_params.
        """
        if update_brush_params:
            self._update_brush_params(y=y, x=y, is_painting=False)
        self._brush.Reset()
        self._brush.NewStroke()
        self._brush.StrokeTo(x, y, 0.0, self.DTIME)

    def _reset_brush_params(self):
        """
        Resetea los parámetros y el objeto brush a sus valores iniciales.
        """
        red, green, blue = (0.0, 0.0, 0.0) if self._background == 'white' else (1.0, 1.0, 1.0)
        hue, saturation, value = utils.rgb_to_hsv(red, green, blue)

        pressure = 0.0 if self._use_pressure else 1
        self._brush_params = collections.OrderedDict([
            ("y", 0.0),
            ("x", 0.0),
            ("pressure", pressure),
            ("log_size", self._log_brush_sizes[0]),
            ("hue", hue),
            ("saturation", saturation),
            ("value", value),
            ("is_painting", False)])
        self._prev_brush_params = None

        # Reset the libmypaint brush object.
        self._move_to(0.0, 0.0, update_brush_params=False)
        self._update_libmypaint_brush(**self._brush_params)

    def reset(self):
        """
        Resetea el envioronment, volviéndolo a su estado inicial.
        """
        self._surface.Clear()
        self._reset_brush_params()

        self.stats = {"total_strokes": 0, "total_disjoint": 0}

        self._episode_step = 0

    def _bezier_to(self, y_c, x_c, y_e, x_e, pressure,
                   log_size, red, green, blue):
        """
        :param y_c: coordenada y del punto de control
        :param x_c: coordenada x del punto de control
        :param y_e: coordenada y del punto final
        :param x_e: coordenada x del punto final
        :param pressure: presión con la que queremos dibujar la curva.
        :param log_size: tamaño logarítmico con el que queremos dibujar la curva.
        :param red: color red de la curva
        :param green: color green de la curva
        :param blue: color blue de la curva

        Dibuja en el canvas una curva de bezier con la presión, tamaño de brush, color,
        punto de control y punto final indicados. El punto inicial es la posición actual de la brush.

        Dibuja la curva de bezier haciendo self.STROKES_PER_STEP trazos. Empieza haciendo los trazos con
        la presión con la que acabo de hacer la curva anterior, y va cambiando la presión de forma lineal
        hasta llegar a la presión indicada.
        """
        self._update_brush_params(y=y_e, x=x_e, pressure=pressure, log_size=log_size,
                                  red=red, green=green, blue=blue, is_painting=True)

        y_s, x_s, pressure_s = [self._prev_brush_params[k] for k in ["y", "x", "pressure"]]
        pressure_e = pressure

        # Compute point along the Bezier curve.
        p_s = np.array([[y_s, x_s]])
        p_c = np.array([[y_c, x_c]])
        p_e = np.array([[y_e, x_e]])
        points = utils.quadratic_bezier(p_s, p_c, p_e, self.STROKES_PER_STEP + 1)[0]

        # We need to perform this pseudo-stroke at the beginning of the curve
        # so that libmypaint handles the pressure correctly.
        if not self._prev_brush_params["is_painting"]:
            self._brush.StrokeTo(x_s, y_s, pressure_s, self.DTIME)

        for t in range(self.STROKES_PER_STEP):
            alpha = float(t + 1) / self.STROKES_PER_STEP
            pressure = pressure_s * (1. - alpha) + pressure_e * alpha
            self._brush.StrokeTo(points[t + 1][1], points[t + 1][0], pressure, self.DTIME)

    def _grid_to_real(self, location):
        """
        Transforma "coordenadas grid" en "coordenadas canvas".
        """
        return tuple(self._canvas_width * float(c) / self._grid_width for c in location)

    def _process_action(self, action):
        """
        :param action: acción a realizar. Debe estar en el formato especificado por self._action_spec
        :return: los valores location (control point + end point), flag, pressure, log_size, red, green
                 y blue que determinan una acción.
        """
        flag = action["flag"]

        # Get pressure and size.
        if self._use_pressure:
            pressure = self.P_VALUES[action["pressure"]]
        else:
            pressure = 1.0
        if self._use_size:
            log_size = self._log_brush_sizes[action["size"]]
        else:
            log_size = self._log_brush_sizes[0]
        if self._use_color:
            red = self.R_VALUES[action["red"]]
            green = self.G_VALUES[action["green"]]
            blue = self.B_VALUES[action["blue"]]
        else:
            red, green, blue = None, None, None

        # Get locations. NOTE: the order of the coordinates is (y, x).
        locations = [
            np.unravel_index(action[k], (self._grid_width, self._grid_width))
            for k in ["control", "end"]]

        # Convert grid coordinates into full resolution coordinates.
        locations = [
            self._grid_to_real(location) for location in locations]

        return locations, flag, pressure, log_size, red, green, blue

    def step(self, action):
        """Realiza un step sobre el environment"""

        # Comprueba que la acción es válida (tiene las shape y los dtype esperados)
        for k in action.keys():
            self._action_spec[k].validate(action[k])

        locations, flag, pressure, log_size, red, green, blue = self._process_action(action)
        loc_control, loc_end = locations

        # Ambiente para realizar la acción de forma atómica.
        self._surface.BeginAtomic()

        if flag:  # El agente produce una nueva pincelada.
            y_c, x_c = loc_control
            y_e, x_e = loc_end
            self._bezier_to(y_c, x_c, y_e, x_e, pressure, log_size, red, green, blue)

            # Actualizamos las estadísticas.
            self.stats["total_strokes"] += 1
            if not self._prev_brush_params["is_painting"]:
                self.stats["total_disjoint"] += 1
        else:  # Movemos la brush a la localización indicada
            y_e, x_e = loc_end
            self._move_to(y_e, x_e)

        self._surface.EndAtomic()

        self._episode_step += 1