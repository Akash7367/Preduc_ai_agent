"""
generate_sample_video.py
------------------------
Generates a synthetic sports-like video to demonstrate the tracking pipeline.

This creates a realistic crowd scene with 15 moving players on a green pitch.
Used when a real sports video cannot be downloaded.

The video is saved as: sample_sports_video.mp4
"""

import cv2
import numpy as np
import math
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

WIDTH, HEIGHT = 1280, 720
FPS = 25
DURATION_SEC = 60
N_PLAYERS = 15


class Player:
    def __init__(self, pid, team):
        self.pid = pid
        self.team = team
        self.x = random.uniform(100, WIDTH - 100)
        self.y = random.uniform(100, HEIGHT - 100)
        # Velocity
        self.vx = random.uniform(-3, 3)
        self.vy = random.uniform(-3, 3)
        # Player size (simulate perspective: closer = bigger)
        self.w = random.randint(28, 38)
        self.h = random.randint(60, 80)
        # Team colour
        self.colour = (0, 100, 220) if team == 0 else (220, 50, 0)
        self.shorts_colour = (0, 50, 180) if team == 0 else (180, 30, 0)
        # Target position for realistic movement
        self.target_x = random.uniform(100, WIDTH - 100)
        self.target_y = random.uniform(100, HEIGHT - 100)
        self.steps_to_target = 0

    def update(self):
        # Periodically pick new target
        if self.steps_to_target <= 0:
            self.target_x = random.uniform(80, WIDTH - 80)
            self.target_y = random.uniform(80, HEIGHT - 80)
            self.steps_to_target = random.randint(30, 120)

        # Move toward target
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.sqrt(dx * dx + dy * dy) + 1e-6
        speed = random.uniform(1.5, 4.5)
        self.vx = 0.7 * self.vx + 0.3 * (dx / dist * speed)
        self.vy = 0.7 * self.vy + 0.3 * (dy / dist * speed)

        self.x += self.vx
        self.y += self.vy
        self.steps_to_target -= 1

        # Bounce off edges
        if self.x < 80:
            self.x = 80; self.vx = abs(self.vx)
        if self.x > WIDTH - 80:
            self.x = WIDTH - 80; self.vx = -abs(self.vx)
        if self.y < 80:
            self.y = 80; self.vy = abs(self.vy)
        if self.y > HEIGHT - 80:
            self.y = HEIGHT - 80; self.vy = -abs(self.vy)

    def draw(self, frame):
        cx, cy = int(self.x), int(self.y)
        hw, hh = self.w // 2, self.h // 2

        # Shadow
        cv2.ellipse(frame, (cx, cy + hh + 4), (hw, 6), 0, 0, 360, (30, 80, 30), -1)

        # Legs (two rectangles)
        leg_w = hw // 2 - 1
        cv2.rectangle(frame, (cx - leg_w - 2, cy + hh // 2), (cx - 1, cy + hh + 2), (20, 20, 20), -1)
        cv2.rectangle(frame, (cx + 1, cy + hh // 2), (cx + leg_w + 2, cy + hh + 2), (20, 20, 20), -1)

        # Jersey (torso)
        cv2.rectangle(frame, (cx - hw, cy - hh // 3), (cx + hw, cy + hh // 2), self.colour, -1)
        # Number on jersey
        cv2.putText(frame, str(self.pid), (cx - 6, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Shorts
        cv2.rectangle(frame, (cx - hw, cy + hh // 2), (cx + hw, cy + hh // 2 + 14), self.shorts_colour, -1)

        # Head
        cv2.circle(frame, (cx, cy - hh // 3 - 14), 12, (210, 170, 130), -1)

        # Bounding box (ground truth – for reference)
        # cv2.rectangle(frame, (cx - hw, cy - hh // 2), (cx + hw, cy + hh + 5), (255, 255, 0), 1)


def draw_pitch(frame):
    """Draw a football pitch."""
    # Grass
    frame[:] = (34, 139, 34)

    # Pitch stripes
    stripe_w = 80
    for i in range(WIDTH // stripe_w + 1):
        if i % 2 == 0:
            x0 = i * stripe_w
            x1 = min(x0 + stripe_w, WIDTH)
            cv2.rectangle(frame, (x0, 0), (x1, HEIGHT), (40, 148, 40), -1)

    # Outer boundary
    cv2.rectangle(frame, (60, 40), (WIDTH - 60, HEIGHT - 40), (255, 255, 255), 3)

    # Centre line
    cv2.line(frame, (WIDTH // 2, 40), (WIDTH // 2, HEIGHT - 40), (255, 255, 255), 2)

    # Centre circle
    cv2.circle(frame, (WIDTH // 2, HEIGHT // 2), 80, (255, 255, 255), 2)
    cv2.circle(frame, (WIDTH // 2, HEIGHT // 2), 4, (255, 255, 255), -1)

    # Penalty areas
    pa_w, pa_h = 180, 280
    # Left
    cv2.rectangle(frame, (60, HEIGHT // 2 - pa_h // 2), (60 + pa_w, HEIGHT // 2 + pa_h // 2), (255, 255, 255), 2)
    # Right
    cv2.rectangle(frame, (WIDTH - 60 - pa_w, HEIGHT // 2 - pa_h // 2), (WIDTH - 60, HEIGHT // 2 + pa_h // 2), (255, 255, 255), 2)

    # Goals
    goal_h = 120
    # Left goal
    cv2.rectangle(frame, (40, HEIGHT // 2 - goal_h // 2), (60, HEIGHT // 2 + goal_h // 2), (255, 255, 255), 3)
    # Right goal
    cv2.rectangle(frame, (WIDTH - 60, HEIGHT // 2 - goal_h // 2), (WIDTH - 40, HEIGHT // 2 + goal_h // 2), (255, 255, 255), 3)


def generate_video(output_path="sample_sports_video.mp4"):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, FPS, (WIDTH, HEIGHT))

    # Create players: team 0 (blue) and team 1 (red) + 1 goalkeeper each
    players = []
    for i in range(N_PLAYERS):
        team = 0 if i < N_PLAYERS // 2 else 1
        p = Player(i + 1, team)
        players.append(p)

    total_frames = DURATION_SEC * FPS
    pitch = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    draw_pitch(pitch)

    print(f"Generating {total_frames} frames ({DURATION_SEC}s @ {FPS}fps)...")

    for frame_idx in range(total_frames):
        frame = pitch.copy()

        # Update and draw players (sort by y for depth ordering)
        players.sort(key=lambda p: p.y)
        for player in players:
            player.update()
            player.draw(frame)

        # Draw scoreboard HUD
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (WIDTH, 40), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        time_str = f"{frame_idx // (FPS * 60):02d}:{(frame_idx // FPS) % 60:02d}"
        cv2.putText(frame, f"LIVE  {time_str}  TEAM A 0 - 0 TEAM B",
                    (WIDTH // 2 - 160, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(frame)

        if frame_idx % (FPS * 5) == 0:
            print(f"  {frame_idx}/{total_frames} frames written")

    writer.release()
    print(f"Video saved: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_video()
