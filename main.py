'''
Author: niconielsen32
Date: 21.12.2021
Title: Real-Time Head Pose Estimation FaceMesh with MediaPipe and OpenCV: A Comprehensive Guide
Source: https://github.com/niconielsen32/ComputerVision/blob/master/headPoseEstimation.py

import os
import numpy as np
import cv2
import mediapipe as mp
import time
import csv
from datetime import datetime

# Prompt for User ID
user_id = input("Enter User ID: ")

# Create a directory named after the user ID
output_dir = os.path.join(os.getcwd(), user_id)
os.makedirs(output_dir, exist_ok=True)

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(128, 0, 128), thickness=2, circle_radius=1)

# Initialize Video Capture
cap = cv2.VideoCapture(0)

# Replace video capture from webcam with video file reading
# video_file = 'video.mp4'
# cap = cv2.VideoCapture(video_file)

# Get frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(os.path.join(output_dir, f'Participant_{user_id}_output.avi'), fourcc, 20.0, (frame_width, frame_height))

# Dictionary to track pose counts with user ID included
pose_counts = {"Looking Left": 0, "Looking Right": 0, "Looking Up": 0, "Looking Down": 0, "Forward": 0}

# List to store pose direction, timestamp, and user ID
pose_timestamps = []

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    start = time.time()

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)  # flipped for selfie view
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_2d = []
    face_3d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [33, 263, 1, 61, 291, 199]:  # Selected landmark points
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])
            distortion_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)

            rmat, _ = cv2.Rodrigues(rotation_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            x_angle = angles[0] * 360
            y_angle = angles[1] * 360
            z_angle = angles[2] * 360

            if y_angle < -10:
                text = "Looking Left"
            elif y_angle > 10:
                text = "Looking Right"
            elif x_angle > 10:
                text = "Looking Up"
            elif x_angle < -10:
                text = "Looking Down"
            else:
                text = "Forward"

            # Update pose counts
            pose_counts[text] += 1

            # Save pose direction, timestamp, and user ID with milliseconds
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            pose_timestamps.append({"UserID": user_id, "Pose": text, "Timestamp": timestamp})

            nose_3d_projection, _ = cv2.projectPoints(nose_3d, rotation_vec, translation_vec, cam_matrix, distortion_matrix)
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y_angle * 10), int(nose_2d[1] - x_angle * 10))

            cv2.line(image, p1, p2, (255, 0, 0), 3)
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, "x: " + str(np.round(x_angle, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "y: " + str(np.round(y_angle, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "z: " + str(np.round(z_angle, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime
        print("FPS: ", fps)
        cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        mp_drawing.draw_landmarks(image=image,
                                  landmark_list=face_landmarks,
                                  connections=mp_face_mesh.FACEMESH_CONTOURS,
                                  landmark_drawing_spec=drawing_spec,
                                  connection_drawing_spec=drawing_spec)

    # Write the frame into the file 'output.avi'
    out.write(image)

    cv2.imshow("Head Pose", image)
    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
out.release()
cv2.destroyAllWindows()


# Write pose data to CSV
def save_pose_timestamps_data():
    with open(os.path.join(output_dir, f'Participant_{user_id}_pose_timestamps.csv'), 'w', newline='') as csvfile:
        fieldnames = ['UserID', 'Pose', 'Timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for entry in pose_timestamps:
            writer.writerow(entry)
    print("Pose timestamps saved to pose_timestamps.csv")


def save_pose_count_data():
    with open(os.path.join(output_dir, f'Participant_{user_id}_pose_counts.csv'), 'w', newline='') as csvfile:
        fieldnames = ['UserID', 'Pose', 'Count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        # writer.writerow({'UserID': user_id})  # Write user ID once at the top
        for pose, count in pose_counts.items():
           writer.writerow({'UserID': user_id, 'Pose': pose, 'Count': count})

    print("Pose counts saved to pose_counts.csv")


save_pose_timestamps_data()
save_pose_count_data()
'''

from flask import Flask, render_template, Response, request, send_file
import cv2
import mediapipe as mp
import csv
import os
import numpy as np
from datetime import datetime
import time
import zipfile

app = Flask(__name__)

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(128, 0, 128), thickness=2, circle_radius=1)

pose_counts = {}
pose_timestamps = []
user_id = None
cap = None
out = None


def generate_frames():
    global cap, face_mesh, user_id, out, pose_counts, pose_timestamps
    if not cap:
        return

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        start = time.time()  # Use time to calculate FPS

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)  # flipped for selfie view
        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_2d = []
        face_3d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [33, 263, 1, 61, 291, 199]:  # Selected landmark points
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])
                distortion_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)

                rmat, _ = cv2.Rodrigues(rotation_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                x_angle = angles[0] * 360
                y_angle = angles[1] * 360
                z_angle = angles[2] * 360

                if y_angle < -10:
                    text = "Looking Left"
                elif y_angle > 10:
                    text = "Looking Right"
                elif x_angle > 10:
                    text = "Looking Up"
                elif x_angle < -10:
                    text = "Looking Down"
                else:
                    text = "Forward"

                # Update pose counts
                if text not in pose_counts:
                    pose_counts[text] = 0
                pose_counts[text] += 1

                # Save pose direction, timestamp, and user ID with milliseconds
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                pose_timestamps.append({"UserID": user_id, "Pose": text, "Timestamp": timestamp})

                nose_3d_projection, _ = cv2.projectPoints(nose_3d, rotation_vec, translation_vec, cam_matrix,
                                                          distortion_matrix)
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y_angle * 10), int(nose_2d[1] - x_angle * 10))

                cv2.line(image, p1, p2, (255, 0, 0), 3)
                cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                cv2.putText(image, "x: " + str(np.round(x_angle, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
                cv2.putText(image, "y: " + str(np.round(y_angle, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
                cv2.putText(image, "z: " + str(np.round(z_angle, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)

            mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, drawing_spec, drawing_spec)

        # Write the frame into the file 'output.avi'
        if out:
            out.write(image)

        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime
        cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/', methods=['GET', 'POST'])
def index():
    global user_id, cap, out, pose_counts, pose_timestamps
    if request.method == 'POST':
        if request.form.get('action') == 'Start Webcam':
            user_id = request.form.get('user_id')
            if user_id:
                # Create a directory for the user ID
                output_dir = os.path.join(os.getcwd(), user_id)
                os.makedirs(output_dir, exist_ok=True)

                # Initialize Video Capture and VideoWriter
                cap = cv2.VideoCapture(0)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(os.path.join(output_dir, f'Participant_{user_id}_output.avi'), fourcc, 20.0,
                                      (frame_width, frame_height))

                # Reset pose data
                pose_counts = {"Looking Left": 0, "Looking Right": 0, "Looking Up": 0, "Looking Down": 0, "Forward": 0}
                pose_timestamps = []

    return render_template('index.html', user_id=user_id)


@app.route('/video_feed')
def video_feed():
    if cap:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        # Display a default image when webcam is not active
        default_image_path = 'static/default_image.jpg'
        if os.path.exists(default_image_path):
            return Response(open(default_image_path, 'rb').read(), mimetype='image/jpeg')
        return "Default image not found.", 404

@app.route('/download_pose_timestamps/<user_id>')
def download_pose_timestamps(user_id):
    file_path = os.path.join(user_id, f'Participant_{user_id}_pose_timestamps.csv')
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "File not found.", 404


@app.route('/download_pose_counts/<user_id>')
def download_pose_counts(user_id):
    file_path = os.path.join(user_id, f'Participant_{user_id}_pose_counts.csv')
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "File not found.", 404

@app.route('/download_csv_zip/<user_id>')
def download_csv_zip(user_id):
    zip_file_path = os.path.join(user_id, f'Participant_{user_id}_data.zip')

    # Create a ZIP file
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        pose_timestamps_file = os.path.join(user_id, f'Participant_{user_id}_pose_timestamps.csv')
        pose_counts_file = os.path.join(user_id, f'Participant_{user_id}_pose_counts.csv')

        # Add files to the ZIP file
        if os.path.exists(pose_timestamps_file):
            zipf.write(pose_timestamps_file, os.path.basename(pose_timestamps_file))
        if os.path.exists(pose_counts_file):
            zipf.write(pose_counts_file, os.path.basename(pose_counts_file))

    # Return the ZIP file as a response
    if os.path.exists(zip_file_path):
        return send_file(zip_file_path, as_attachment=True)
    return "File not found.", 404

@app.route('/stop')
def stop():
    global cap, out, user_id, pose_counts, pose_timestamps
    if cap:
        cap.release()
        cap = None
    if out:
        out.release()
        out = None

    if user_id:
        # Save data to CSV files
        save_pose_timestamps_data(user_id)
        save_pose_count_data(user_id)

        # Create and save ZIP file
        zip_file_path = os.path.join(user_id, f'Participant_{user_id}_data.zip')
        with zipfile.ZipFile(zip_file_path, 'w') as zipf:
            pose_timestamps_file = os.path.join(user_id, f'Participant_{user_id}_pose_timestamps.csv')
            pose_counts_file = os.path.join(user_id, f'Participant_{user_id}_pose_counts.csv')

            if os.path.exists(pose_timestamps_file):
                zipf.write(pose_timestamps_file, os.path.basename(pose_timestamps_file))
            if os.path.exists(pose_counts_file):
                zipf.write(pose_counts_file, os.path.basename(pose_counts_file))

    return 'Stopped webcam and tracking.'



def save_pose_timestamps_data(user_id):
    file_path = os.path.join(user_id, f'Participant_{user_id}_pose_timestamps.csv')
    with open(file_path, 'w', newline='') as csvfile:
        fieldnames = ['UserID', 'Pose', 'Timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in pose_timestamps:
            writer.writerow(entry)


def save_pose_count_data(user_id):
    file_path = os.path.join(user_id, f'Participant_{user_id}_pose_counts.csv')
    with open(file_path, 'w', newline='') as csvfile:
        fieldnames = ['UserID', 'Pose', 'Count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for pose, count in pose_counts.items():
            writer.writerow({'UserID': user_id, 'Pose': pose, 'Count': count})


if __name__ == '__main__':
    app.run(debug=True)

