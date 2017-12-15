class ProgressBar:
  def __init__(self, units=30, maxval=100):
    self.units = units
    self.maxval = maxval
    print("")

  def printProgress(self, val, prefix="Progress", suffix=""):
    progress = (val + 1) * self.units // self.maxval
    progbar = "\r\033[F" + prefix + ": ["
    for i in range(progress):
      progbar += "="
    if progress != self.units:
      progbar += ">"
    for i in range(self.units - progress - 1):
      progbar += " "
    progbar += "] " + suffix
    print(progbar)
