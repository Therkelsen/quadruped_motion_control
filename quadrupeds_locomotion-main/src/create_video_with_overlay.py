import cv2
import numpy as np
import pickle

def normalize_commands(commands_buffer):
    max_lin_x = max(abs(cmd[0]) for cmd in commands_buffer)
    max_lin_y = max(abs(cmd[1]) for cmd in commands_buffer)
    max_ang_z = max(abs(cmd[2]) for cmd in commands_buffer)
    max_base_height = max(abs(cmd[3]) for cmd in commands_buffer)
    max_jump_height = max(abs(cmd[4]) for cmd in commands_buffer)
    return max_lin_x, max_lin_y, max_ang_z, max_base_height, max_jump_height

def draw_joystick(image, lin_x, lin_y, ang_z, base_height, jump_height, max_lin_x, max_lin_y, radius=100, x_offset=10, y_offset=10):
    # Draw the joystick base with gradient directly on the image
    for i in range(radius):
        r = radius - i
        # color = (255, int(255 * (0.5 + 0.5 * i / radius)), int(255 * (0.5 + 0.5 * i / radius)))
        color = (int(55+200 * (0.5 + 0.5 * i / radius)), int(55+200 * (0.5 + 0.5 * i / radius)), int(55+200 * (0.5 + 0.5 * i / radius)))
        cv2.circle(image, (x_offset + radius, y_offset + radius), r, color, -1)

    # Draw the joystick position with shadow directly on the image
    joystick_x = int(x_offset + radius + (lin_y / max_lin_y) * radius)
    joystick_y = int(y_offset + radius - (lin_x / max_lin_x) * radius)
    cv2.circle(image, (joystick_x + 2, joystick_y + 2), int(radius * 0.12), (0, 0, 0), -1)  # Shadow
    # cv2.circle(image, (joystick_x, joystick_y), int(radius * 0.1), (0, 0, 255), -1)

    return image


def draw_target_height_bar(image, base_height, max_base_height, target_height=1.0, x_offset=220, y_offset=10):
    base_height = max(0, base_height)  # Ensure base_height is non-negative
    # Create a bar to represent the target height
    bar_width = 20
    bar_height = 200
    bar_x = x_offset  # Place the bar to the right of the joystick
    bar_y = y_offset  # Align with the top of the joystick

    # Draw the background of the bar
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), -1)

    # Draw the current height indicator
    current_height_pos = int(bar_y + bar_height - (base_height / max_base_height) * bar_height)
    cv2.rectangle(image, (bar_x, current_height_pos), (bar_x + bar_width, bar_y + bar_height), (0, 255, 0), -1)

    # Draw the target height line
    target_height_pos = int(bar_y + bar_height - (target_height / target_height) * bar_height)
    cv2.line(image, (bar_x, target_height_pos), (bar_x + bar_width, target_height_pos), (0, 0, 255), 2)

    return image

def draw_angular_velocity_bar(image, ang_z, max_ang_z, x_offset=10, y_offset=220):
    # Create a bar to represent the angular velocity
    bar_width = 200
    bar_height = 20
    bar_x = x_offset  # Use the provided x_offset
    bar_y = y_offset  # Use the provided y_offset

    # Draw the background of the bar
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), -1)

    # Draw the current angular velocity indicator
    current_ang_pos = int(bar_x + (ang_z / max_ang_z + 1) / 2 * bar_width)  # Normalize ang_z to [0, 1]
    cv2.rectangle(image, (bar_x, bar_y), (current_ang_pos, bar_y + bar_height), (0, 255, 0), -1)

    return image

def create_video_with_overlay(images_buffer, commands_buffer, output_video_path, fps=30):
    # Get the dimensions of the images
    height, width, _ = images_buffer[0].shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    max_lin_x, max_lin_y, max_ang_z, max_base_height, max_jump_height = normalize_commands(commands_buffer)

    for i in range(len(images_buffer)):
        image = images_buffer[i]
        # Invert the image channels (RGB to BGR) for OpenCV
        
        lin_x, lin_y, ang_z, base_height, jump_height = commands_buffer[i]

        
        # Overlay the joystick on the image (top-left corner)
        x_offset = images_buffer[0].shape[1] // 2 - 100
        y_offset = images_buffer[0].shape[0]  - 250
        radius = 100
        
        # Draw the joystick overlay
        image = draw_joystick(image, lin_x, lin_y, ang_z, base_height, jump_height, max_lin_x, max_lin_y, radius=radius, x_offset=x_offset, y_offset=y_offset)


        # Draw the target height bar
        image = draw_target_height_bar(image, base_height, max_base_height, x_offset=x_offset + radius*2 + 10, y_offset=y_offset)

        # Draw the angular velocity bar with adjusted position
        image = draw_angular_velocity_bar(image, ang_z, max_ang_z, x_offset=x_offset, y_offset=y_offset + radius*2 + 20)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Write the frame to the video
        out.write(image)

    # Release the VideoWriter
    out.release()

# Load the images and commands from the pickle files
images_buffer = pickle.load(open("images_buffer.pkl", "rb"))
commands_buffer = pickle.load(open("commands_buffer.pkl", "rb"))

# Create the video with the joystick overlay and target height bar
output_video_path = "output_video.mp4"
create_video_with_overlay(images_buffer, commands_buffer, output_video_path, fps=30)

print(f"Video saved to {output_video_path}")