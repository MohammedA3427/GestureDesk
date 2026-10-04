# GestureDesk

A local, real-time webcam interface for the Ontario Tech Reverse Career Fair. 

## Start Instructions

Double-click **Start-GestureDesk.cmd**. 

Double-click **Start-Keyboard-Demo.cmd** for the presentation.

## Demo controls

| Input | Action |
|---|---|
| Open palm moving horizontally | Next/previous portfolio card in the direction shown on screen |
| Peace sign held about 220 ms | Toggle air drawing; release/change pose before repeating |
| Thumb/index pinch with other fingers extended | Draw with index fingertip in drawing mode |
| D | Toggle canvas |
| Left/right arrows | Browse portfolio |
| Space | Pause/resume interaction |
| C | Clear drawing (keyboard only, to avoid accidental erasure) |
| H | Toggle landmark skeleton |
| Q / Esc | Quit |

Point, fist and thumbs-up are recognized for inspection but have no destructive actions. Hold one hand fully in view. Two visible hands suspend recognition. Swipes require an open palm, mostly horizontal monotonic travel, and have an 800 ms cooldown. Peace toggles once per stable hold; a brief recognition flicker cannot repeatedly toggle it.