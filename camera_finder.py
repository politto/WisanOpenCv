import cv2
from pyusbcameraindex import enumerate_usb_video_devices_windows

# List the devices.
devices = enumerate_usb_video_devices_windows()
for device in devices:
    print(f"{device.index} == {device.name} (VID: {device.vid}, PID: {device.pid}, Path: {device.path}")

# Show a frame from each.
for device in devices:
    cap = cv2.VideoCapture(device.index, cv2.CAP_DSHOW)
    ret, frame = cap.read()
    cv2.imshow(f"Device={device.name}", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cap.release()
cv2.destroyAllWindows()
