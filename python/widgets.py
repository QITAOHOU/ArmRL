import pygame

class Slider():
  def __init__(self, value_range, pos, shape, buttonColor=(0x03, 0xA9, 0xF4),
      fillColor=(0xF5, 0xF5, 0xF5), lineColor=(0xBD, 0xBD, 0xBD), radius=None,
      initialValue=0):
    self.value_range = value_range
    self.pos = pos
    self.shape = shape # (width, height)
    self.buttonColor = buttonColor
    self.pad = min(self.shape[0], self.shape[1])
    self.button_render = pygame.Surface((self.pad, self.pad)).convert_alpha()
    self.button_render.fill((0, 0, 0, 0))
    self.radius = radius
    if radius == None:
      self.radius = self.pad / 3
    pygame.draw.circle(self.button_render, self.buttonColor,
        (self.pad // 2, self.pad // 2), self.radius)
    self.length = max(self.shape[0], self.shape[1]) - \
        min(self.shape[0], self.shape[1])
    self.slide_render = pygame.Surface(self.shape)
    self.slide_render.fill(fillColor)
    if self.shape[1] > self.shape[0]:
      pygame.draw.line(self.slide_render, lineColor,
          (self.pad // 2, self.pad // 2),
          (self.pad // 2, self.pad // 2 + self.length), 3)
    else:
      pygame.draw.line(self.slide_render, lineColor,
          (self.pad // 2, self.pad // 2),
          (self.pad // 2 + self.length, self.pad // 2), 3)
    self.value = initialValue
    self.reversed = False
    self.active = False

  def render(self, surface):
    surface.blit(self.slide_render, self.pos)
    ratio = (self.value - self.value_range[0]) / \
        (self.value_range[1] - self.value_range[0])
    if self.reversed:
      ratio = 1.0 - ratio
    offset = int(ratio * self.length)
    highlight = pygame.Surface(self.shape).convert_alpha()
    highlight.fill((0, 0, 0, 0))
    if self.shape[1] > self.shape[0]:
      pygame.draw.line(highlight, self.buttonColor,
          (self.pad // 2, self.pad // 2),
          (self.pad // 2, self.pad // 2 + offset), 3)
      surface.blit(highlight, self.pos)
      surface.blit(self.button_render, (self.pos[0], self.pos[1] + offset))
    else:
      pygame.draw.line(highlight, self.buttonColor,
          (self.pad // 2, self.pad // 2),
          (self.pad // 2 + offset, self.pad // 2), 3)
      surface.blit(highlight, self.pos)
      surface.blit(self.button_render, (self.pos[0] + offset, self.pos[1]))

  def hovered(self, mousepos):
    return mousepos[0] >= self.pos[0] and \
           self.mousepos[0] < self.pos[0] + self.shape[0] and \
           mousepos[1] >= self.pos[1] and \
           self.mousepos[1] < self.pos[1] + self.shape[1]

  def inActiveRegion(self, mousepos):
    clickpad = self.pad / 6
    return mousepos[0] >= self.pos[0] + clickpad and \
           mousepos[0] < self.pos[0] + self.shape[0] - clickpad and \
           mousepos[1] >= self.pos[1] + clickpad and \
           mousepos[1] < self.pos[1] + self.shape[1] - clickpad

  def update(self, mousepos):
    if not self.active:
      return
    truncated_pos = 0
    if self.shape[1] > self.shape[0]:
      startpos = self.pad // 2 + self.pos[1]
      mouse = mousepos[1]
    else:
      startpos = self.pad // 2 + self.pos[0]
      mouse = mousepos[0]
    abs_pos = min(startpos + self.length, max(startpos, mouse))
    rel_pos = abs_pos - startpos
    ratio = rel_pos / self.length
    if self.reversed:
      ratio = 1.0 - ratio
    self.value = ratio * (self.value_range[1] - self.value_range[0]) + \
        self.value_range[0]

  def getValue(self):
    return self.value

  def setActive(self, mousepos):
    if self.inActiveRegion(mousepos):
      self.active = True

  def setInactive(self):
    self.active = False

class TextBox():
  def __init__(self, pos, shape, fontColor=(0x03, 0xA9, 0xF4),
      fillColor=(0xF5, 0xF5, 0xF5), initialValue="0", fmt="%s"):
    self.pos = pos
    self.shape = shape
    self.fontColor = fontColor
    self.fillColor = fillColor
    self.font = pygame.font.SysFont("Calibri", int(shape[1] * 2 / 3))
    self.value = initialValue
    self.fmt = fmt

  def setValue(self, value):
    self.value = value

  def render(self, surface):
    text = self.font.render(self.fmt % self.value, True, self.fontColor)
    textArea = pygame.Surface(self.shape)
    textArea.fill(self.fillColor)
    textArea.blit(text, ((self.shape[0] - text.get_width()) // 2, \
                         (self.shape[1] - text.get_height()) // 2))
    surface.blit(textArea, self.pos)

  def hovered(self, mousepos):
    return mousepos[0] >= self.pos[0] and \
           self.mousepos[0] < self.pos[0] + self.shape[0] and \
           mousepos[1] >= self.pos[1] and \
           self.mousepos[1] < self.pos[1] + self.shape[1]

class SliderCounter():
  def __init__(self, value_range, pos, shape, radius=None, counterWidth=None,
      fmt="%s", initialValue=0):
    if counterWidth == None:
      counterWidth = shape[1]
    self.slider = Slider(value_range, pos, (shape[0] - counterWidth, shape[1]),
        radius=radius, initialValue=initialValue)
    self.counter = TextBox((shape[0] - counterWidth + pos[0], pos[1]),
        (counterWidth, shape[1]), fmt=fmt, initialValue=initialValue)

  def render(self, surface):
    self.counter.setValue(self.slider.getValue())
    self.slider.render(surface)
    self.counter.render(surface)

  def getValue(self):
    return self.slider.getValue()

  def setActive(self, mousepos):
    return self.slider.setActive(mousepos)

  def setInactive(self):
    self.slider.setInactive()

  def update(self, mousepos):
    self.slider.update(mousepos)

  def inActiveRegion(self, mousepos):
    return self.slider.inActiveRegion(mousepos)

  def hovered(self, mousepos):
    return self.slider.hovered(mousepos) or self.counter.hovered(mousepos)
