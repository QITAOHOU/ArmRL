#!/usr/bin/env python3
import pygame, sys
from widgets import SliderCounter, TextBox
import numpy as np
import physx
import simulation
import argparse

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--printpos", action="store_true",
      help="Print the global positions")
  args = parser.parse_args()

  # connect to the visualizer
  arm = simulation.Arm()

  # create a window for the controls
  pygame.init()
  screen = pygame.display.set_mode((640, 360))
  clock = pygame.time.Clock()

  # place widgets to respresent the controls that the user can input
  labels = []
  linkSlides = []
  jointSlides = []
  for i in range(arm.num_joints):
    labels.append(TextBox((0, 50 * i + 15), (80, 30),
      initialValue="Link " + str(i + 1)))
    linkSlides.append(SliderCounter((0, 10),
      (80, 50 * i + 15), (240, 30), radius=8, counterWidth=60, fmt="%0.1f",
      initialValue=arm.default_length[i]))
    labels.append(TextBox((320, 50 * i + 15), (80, 30),
      initialValue="Joint " + str(i + 1)))
    jointSlides.append(
        SliderCounter((arm.joint_limits[i][0], arm.joint_limits[i][1]), 
          (400, 50 * i + 15), (240, 30), radius=8, counterWidth=60, fmt="%d"))
  
  while True:
    # update the controls events
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        pygame.quit()
        sys.exit()
      elif event.type == pygame.MOUSEBUTTONDOWN:
        for link in linkSlides:
          link.setActive(pygame.mouse.get_pos())
        for joint in jointSlides:
          joint.setActive(pygame.mouse.get_pos())
      elif event.type == pygame.MOUSEMOTION:
        for link in linkSlides:
          link.update(pygame.mouse.get_pos())
        for joint in jointSlides:
          joint.update(pygame.mouse.get_pos())
      elif event.type == pygame.MOUSEBUTTONUP:
        for link in linkSlides:
          link.setInactive()
        for joint in jointSlides:
          joint.setInactive()

    # using the inputs from the controls panel, calculate the forward kinematics
    # to get the positions and set those on the arm
    positions = physx.forwardKinematics(
        np.array([link.getValue() for link in linkSlides]),
        np.array([joint.getValue() for joint in jointSlides]))
    arm.setPositions(positions)
    if args.printpos:
      print(positions)

    # update the controls render
    screen.fill((0xF5, 0xF5, 0xF5))
    for label in labels:
      label.render(screen)
    for link in linkSlides:
      link.render(screen)
    for joint in jointSlides:
      joint.render(screen)
    pygame.display.flip()
    clock.tick(50)

if __name__ == "__main__":
  main()
