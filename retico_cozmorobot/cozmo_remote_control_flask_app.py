#!/usr/bin/env python3

# Copyright (c) 2016 Anki, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License in the file LICENSE.txt or at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Control Cozmo using a webpage on your computer.

This example lets you control Cozmo by Remote Control, using a webpage served by Flask.
'''

import asyncio
import io
import json
import math
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np

from retico_cozmorobot import flask_helpers

import cozmo

from collections import deque

try:
    from flask import Flask, request
except ImportError:
    sys.exit("Cannot import from flask: Do `pip3 install --user flask` to install")

try:
    from PIL import Image, ImageDraw
except ImportError:
    sys.exit("Cannot import from PIL: Do `pip3 install --user Pillow` to install")

try:
    import requests
except ImportError:
    sys.exit("Cannot import from requests: Do `pip3 install --user requests` to install")


DEBUG_ANNOTATIONS_DISABLED = 0
DEBUG_ANNOTATIONS_ENABLED_VISION = 1
DEBUG_ANNOTATIONS_ENABLED_ALL = 2


# Annotator for displaying RobotState (position, etc.) on top of the camera feed
class RobotStateDisplay(cozmo.annotate.Annotator):
    def apply(self, image, scale):
        d = ImageDraw.Draw(image)

        bounds = [3, 0, image.width, image.height]

        def print_line(text_line):
            text = cozmo.annotate.ImageText(text_line, position=cozmo.annotate.TOP_LEFT, outline_color='black', color='lightblue')
            text.render(d, bounds)
            TEXT_HEIGHT = 11
            bounds[1] += TEXT_HEIGHT

        robot = self.world.robot  # type: cozmo.robot.Robot

        # Display the Pose info for the robot

        pose = robot.pose
        print_line('Pose: Pos = <%.1f, %.1f, %.1f>' % pose.position.x_y_z)
        print_line('Pose: Z = <%.2f>' % pose.position.z)
        print_line('Pose: Rot quat = <%.1f, %.1f, %.1f, %.1f>' % pose.rotation.q0_q1_q2_q3)
        print_line('Pose: angle_z = %.1f' % pose.rotation.angle_z.degrees)
        print_line('Pose: origin_id: %s' % pose.origin_id)

        # Display the Accelerometer and Gyro data for the robot

        print_line('Accelmtr: <%.1f, %.1f, %.1f>' % robot.accelerometer.x_y_z)
        print_line('Gyro: <%.1f, %.1f, %.1f>' % robot.gyro.x_y_z)

        # Display the Accelerometer and Gyro data for the mobile device

        if robot.device_accel_raw is not None:
            print_line('Device Acc Raw: <%.2f, %.2f, %.2f>' % robot.device_accel_raw.x_y_z)
        if robot.device_accel_user is not None:
            print_line('Device Acc User: <%.2f, %.2f, %.2f>' % robot.device_accel_user.x_y_z)
        if robot.device_gyro is not None:
            mat = robot.device_gyro.to_matrix()
            print_line('Device Gyro Up: <%.2f, %.2f, %.2f>' % mat.up_xyz)
            print_line('Device Gyro Fwd: <%.2f, %.2f, %.2f>' % mat.forward_xyz)
            print_line('Device Gyro Left: <%.2f, %.2f, %.2f>' % mat.left_xyz)


def create_default_image(image_width, image_height, do_gradient=False):
    '''Create a place-holder PIL image to use until we have a live feed from Cozmo'''
    image_bytes = bytearray([0x70, 0x70, 0x70]) * image_width * image_height

    if do_gradient:
        i = 0
        for y in range(image_height):
            for x in range(image_width):
                image_bytes[i] = int(255.0 * (x / image_width))   # R
                image_bytes[i+1] = int(255.0 * (y / image_height))  # G
                image_bytes[i+2] = 0                                # B
                i += 3

    image = Image.frombytes('RGB', (image_width, image_height), bytes(image_bytes))
    return image


flask_app = Flask(__name__)
remote_control_cozmo = None
_default_camera_image = create_default_image(320, 240)

_is_save_pose_btn_enabled = True

_display_debug_annotations = DEBUG_ANNOTATIONS_ENABLED_ALL


def remap_to_range(x, x_min, x_max, out_min, out_max):
    '''convert x (in x_min..x_max range) to out_min..out_max range'''
    if x < x_min:
        return out_min
    elif x > x_max:
        return out_max
    else:
        ratio = (x - x_min) / (x_max - x_min)
        return out_min + ratio * (out_max - out_min)


class RemoteControlCozmo:

    def __init__(self, coz):
        self.cozmo = coz

        self.drive_forwards = 0
        self.drive_back = 0
        self.turn_left = 0
        self.turn_right = 0
        self.lift_up = 0
        self.lift_down = 0
        self.head_up = 0
        self.head_down = 0

        self.go_fast = 0
        self.go_slow = 0

        self.action_queue = []
        self.text_to_say = "Hi I'm Cozmo"
        self.pose_queue = deque(maxlen=1)
        self.all_buttons_disabled = True
    def enable_savepose_button(self):
        global _is_save_pose_btn_enabled
        _is_save_pose_btn_enabled = True

    def disable_savepose_button(self):
        global _is_save_pose_btn_enabled
        _is_save_pose_btn_enabled = False

    def run(self, remote_control_coz):
        global remote_control_cozmo
        remote_control_cozmo = remote_control_coz
        remote_control_coz.cozmo.world.image_annotator.add_annotator('robotState', RobotStateDisplay)
        flask_helpers.run_flask(flask_app, host_port=8112)
        # threading.Thread(target=flask_helpers.run_flask, args=[flask_app, "127.0.0.1", 8112]).start()
        # threading.Thread(target=lambda: flask_helpers.run_flask(flask_app, host_port=8112)).start()



    def get_robot_pose(self):
        pose = self.cozmo.pose
        self.pose_queue.append(pose)
        return pose


    def handle_key(self, key_code, is_shift_down, is_ctrl_down, is_alt_down, is_key_down):
        '''Called on any key press or release
           Holding a key down may result in repeated handle_key calls with is_key_down==True
        '''

        # Update desired speed / fidelity of actions based on shift/alt being held
        was_go_fast = self.go_fast
        was_go_slow = self.go_slow

        self.go_fast = is_shift_down
        self.go_slow = is_alt_down

        speed_changed = (was_go_fast != self.go_fast) or (was_go_slow != self.go_slow)

        # Update state of driving intent from keyboard, and if anything changed then call update_driving
        update_driving = True
        if key_code == ord('W'):
            self.drive_forwards = is_key_down
        elif key_code == ord('S'):
            self.drive_back = is_key_down
        elif key_code == ord('A'):
            self.turn_left = is_key_down
        elif key_code == ord('D'):
            self.turn_right = is_key_down
        else:
            if not speed_changed:
                update_driving = False

        # Update state of lift move intent from keyboard, and if anything changed then call update_lift
        update_lift = True
        if key_code == ord('R'):
            self.lift_up = is_key_down
        elif key_code == ord('F'):
            self.lift_down = is_key_down
        else:
            if not speed_changed:
                update_lift = False

        # Update state of head move intent from keyboard, and if anything changed then call update_head
        update_head = True
        if key_code == ord('T'):
            self.head_up = is_key_down
        elif key_code == ord('G'):
            self.head_down = is_key_down
        else:
            if not speed_changed:
                update_head = False

        # Update driving, head and lift as appropriate
        if update_driving: # This says mouse driving but includes wasd driving
            self.update_driving()
        if update_head:
            self.update_head()
        if update_lift:
            self.update_lift()

        # Handle any keys being released (e.g. the end of a key-click)
        if not is_key_down:
            if key_code == ord(' '):
                self.say_text(self.text_to_say)

    def func_to_name(self, func):
        if func == self.try_say_text:
            return "say_text"
        else:
            return "UNKNOWN"


    def action_to_text(self, action):
        func, args = action
        return self.func_to_name(func) + "( " + str(args) + " )"


    def action_queue_to_text(self, action_queue):
        out_text = ""
        i = 0
        for action in action_queue:
            out_text += "[" + str(i) + "] " + self.action_to_text(action)
            i += 1
        return out_text


    def queue_action(self, new_action):
        if len(self.action_queue) > 10:
            self.action_queue.pop(0)
        self.action_queue.append(new_action)


    def try_say_text(self, text_to_say):
        try:
            self.cozmo.say_text(text_to_say)
            return True
        except cozmo.exceptions.RobotBusy:
            return False


    def say_text(self, text_to_say):
        self.queue_action((self.try_say_text, text_to_say))
        self.update()

    def update(self):
        '''Try and execute the next queued action'''
        if len(self.action_queue) > 0:
            queued_action, action_args = self.action_queue[0]
            if queued_action(action_args):
                self.action_queue.pop(0)

    def pick_speed(self, fast_speed, mid_speed, slow_speed):
        if self.go_fast:
            if not self.go_slow:
                return fast_speed
        elif self.go_slow:
            return slow_speed
        return mid_speed


    def update_lift(self):
        lift_speed = self.pick_speed(8, 4, 2)
        lift_vel = (self.lift_up - self.lift_down) * lift_speed
        self.cozmo.move_lift(lift_vel)


    def update_head(self):
        head_speed = self.pick_speed(2, 1, 0.5)
        head_vel = (self.head_up - self.head_down) * head_speed
        self.cozmo.move_head(head_vel)

    def scale_deadzone(self, value, deadzone, maximum):
        if math.fabs(value) > deadzone:
            adjustment = math.copysign(deadzone, value)
            scaleFactor = maximum / (maximum - deadzone)
            return (value - adjustment) * scaleFactor
        else:
            return 0

    def update_driving(self):
        drive_dir = (self.drive_forwards - self.drive_back)

        if (drive_dir > 0.1) and self.cozmo.is_on_charger:
            # cozmo is stuck on the charger, and user is trying to drive off - issue an explicit drive off action
            try:
                # don't wait for action to complete - we don't want to block the other updates (camera etc.)
                self.cozmo.drive_off_charger_contacts()
            except cozmo.exceptions.RobotBusy:
                # Robot is busy doing another action - try again next time we get a drive impulse
                pass

        turn_dir = (self.turn_right - self.turn_left)
        if drive_dir < 0:
            # It feels more natural to turn the opposite way when reversing
            turn_dir = -turn_dir

        forward_speed = self.pick_speed(150, 75, 50)
        turn_speed = self.pick_speed(100, 50, 30)

        l_wheel_speed = (drive_dir * forward_speed) + (turn_speed * turn_dir)
        r_wheel_speed = (drive_dir * forward_speed) - (turn_speed * turn_dir)

        self.cozmo.drive_wheels(l_wheel_speed, r_wheel_speed, l_wheel_speed*4, r_wheel_speed*4 )

def to_js_bool_string(bool_value):
    return "true" if bool_value else "false"

def toggle_is_save_pose_btn():
    global _is_save_pose_btn_enabled
    _is_save_pose_btn_enabled = not _is_save_pose_btn_enabled
    return _is_save_pose_btn_enabled


@flask_app.route("/")
def handle_index_page():
    return '''
    <html>
        <head>
            <title>remote_control_cozmo.py display</title>
        </head>
        <body>
            <h1>Remote Control Cozmo</h1>
            <table>
                <tr>
                    <td valign = top>
                        <div id="cozmoImageMicrosoftWarning" style="display: none;color: #ff9900; text-align: center;">Video feed performance is better in Chrome or Firefox due to mjpeg limitations in this browser</div>
                        <img src="cozmoImage" id="cozmoImageId" width=640 height=480>
                        <div id="DebugInfoId"></div>
                    </td>
                    <td width=30></td>
                    <td valign=top>
                        <b>Shutdown Server</b> : <button id="shutdownId" onClick=shutdownServerButtonClicked(this) style="font-size: 14px">Shutdown</button><br>
                        <b>Save Robot Pose</b> : <button id="saveRobotPoseId" onClick=saveRobotPoseClicked(this) style="font-size: 14px">Save Pose</button><br>
                        <h2>Controls:</h2>

                        <h3>Driving:</h3>

                        <b>W A S D</b> : Drive Forwards / Left / Back / Right<br><br>
                        <b>T</b> : Move Head Up<br>
                        <b>G</b> : Move Head Down<br>

                        <h3>Lift:</h3>
                        <b>R</b> : Move Lift Up<br>
                        <b>F</b>: Move Lift Down<br>
                        <h3>General:</h3>
                        <b>Shift</b> : Hold to Move Faster (Driving, Head and Lift)<br>
                        <b>Alt</b> : Hold to Move Slower (Driving, Head and Lift)<br>
                        <b>L</b> : Toggle IR Headlight: <button id="headlightId" onClick=onHeadlightButtonClicked(this) style="font-size: 14px">Default</button><br>
                        <b>O</b> : Toggle Debug Annotations: <button id="debugAnnotationsId" onClick=onDebugAnnotationsButtonClicked(this) style="font-size: 14px">Default</button><br>
                        <b>Z</b> : Toggle Camera Color mode: <button id="cameraColorId" onClick=onCameraColorButtonClicked(this) style="font-size: 14px">Save Image</button><br>
                        <h3>Talk</h3>
                        <b>Space</b> : Say <input type="text" name="sayText" id="sayTextId" value="''' + remote_control_cozmo.text_to_say + '''" onchange=handleTextInput(this)>
                        
                        <h3>Save Image</h3>
                        <b>Save</b> : Save image from current camera view: 
                        <input type="text" name="saveImageFileName" id="saveImageTextId" value="image_name.png">
                        <button id="saveImageButtonId" onClick=saveImageClicked(saveImageTextId) style="font-size: 14px">Save Image</button><br>
                        
                        <h3>Adjust Camera</h3>
                        <b>Save</b> : Adjust camera settings: <br>
                        <input type="text" name="adjustGain" id="adjustGainId" value="0.05">
                        <button id="saveGainSettingsButtonId" onClick=saveGainSettingsClicked(adjustGainId) style="font-size: 14px">Save Gain</button><br>

                        <input type="text" name="adjustExposure" id="adjustExposureId" value="0.5">
                        <button id="saveExposureSettingsButtonId" onClick=saveExposureSettingsClicked(adjustExposureId) style="font-size: 14px">Save Exposure</button><br>
                        
                    </td>
                </tr>
            </table>

            <script type="text/javascript">
                var gLastClientX = -1
                var gLastClientY = -1
                var gIsSavePoseEnabled = ''' + to_js_bool_string(_is_save_pose_btn_enabled) + '''
                var gAreDebugAnnotationsEnabled = '''+ str(_display_debug_annotations) + '''
                var gIsHeadlightEnabled = false
                var gIsCameraColorEnabled = true
                var gUserAgent = window.navigator.userAgent;
                var gIsMicrosoftBrowser = gUserAgent.indexOf('MSIE ') > 0 || gUserAgent.indexOf('Trident/') > 0 || gUserAgent.indexOf('Edge/') > 0;
                var gSkipFrame = false;
                var intervalId;

                if (gIsMicrosoftBrowser) {
                    document.getElementById("cozmoImageMicrosoftWarning").style.display = "block";
                }
                
                 function postHttpRequest(url, dataSet)
                {
                    var xhr = new XMLHttpRequest();
                    xhr.open("POST", url, true);
                    xhr.send( JSON.stringify( dataSet ) ); 
                }
                
                function postHttpRequestWithCallback(url, dataSet, callback)
                {
                    var xhr = new XMLHttpRequest();
                    xhr.onreadystatechange = () => {
                      if (xhr.readyState === 4) {
                        callback(xhr.response);
                      }
                    };
                    xhr.open("POST", url, true);
                    xhr.send( JSON.stringify( dataSet ) ); 
                }

                function updateCozmo()
                {
                    if (gIsMicrosoftBrowser && !gSkipFrame) {
                        // IE doesn't support MJPEG, so we need to ping the server for more images.
                        // Though, if this happens too frequently, the controls will be unresponsive.
                        gSkipFrame = true;
                        document.getElementById("cozmoImageId").src="cozmoImage?" + (new Date()).getTime();
                    } else if (gSkipFrame) {
                        gSkipFrame = false;
                    }
                    var xhr = new XMLHttpRequest();
                    xhr.onreadystatechange = function() {
                        if (xhr.readyState == XMLHttpRequest.DONE) {
                            document.getElementById("DebugInfoId").innerHTML = xhr.responseText
                        }
                    }

                    xhr.open("POST", "updateCozmo", true);
                    xhr.send( null );
                    setTimeout(updateCozmo , 60);
                }
                setTimeout(updateCozmo , 60);

                function updateButtonEnabledText(button, isEnabled)
                {
                    button.firstChild.data = isEnabled ? "Enabled" : "Disabled";
                }

                function updateButtonEnabled(button)
                {
                    clearInterval(intervalId);
                    // Disable the button
                    button.disabled = gIsSavePoseEnabled == "true" ? false : true
                    // Loop to determine when the button should be re-enabled (processing has been finished)
                    intervalId = setInterval(() => {
                            postHttpRequestWithCallback("isSavePoseBtnEnabled", '', updateIsSavePoseEnabled)
                            button.disabled = gIsSavePoseEnabled == "true" ? false : true
                      }, 1000);   
                }
                
                function updateDebugAnnotationButtonEnabledText(button, isEnabled)
                {
                    switch(gAreDebugAnnotationsEnabled)
                    {
                    case 0:
                        button.firstChild.data = "Disabled";
                        break;
                    case 1:
                        button.firstChild.data = "Enabled (vision)";
                        break;
                    case 2:
                        button.firstChild.data = "Enabled (all)";
                        break;
                    default:
                        button.firstChild.data = "ERROR";
                        break;
                    }
                }

                function onDebugAnnotationsButtonClicked(button)
                {
                    gAreDebugAnnotationsEnabled += 1;
                    if (gAreDebugAnnotationsEnabled > 2)
                    {
                        gAreDebugAnnotationsEnabled = 0
                    }

                    updateDebugAnnotationButtonEnabledText(button, gAreDebugAnnotationsEnabled)

                    areDebugAnnotationsEnabled = gAreDebugAnnotationsEnabled
                    postHttpRequest("setAreDebugAnnotationsEnabled", {areDebugAnnotationsEnabled})
                }
                
                function shutdownServerButtonClicked(button)
                {
                    postHttpRequest("shutdown")
                }
                function saveRobotPoseClicked(button)
                {
                    postHttpRequestWithCallback("toggleSavePoseBtn", '', updateIsSavePoseEnabled)
                    updateButtonEnabled(button);
                    postHttpRequest("saveRobotPose")
                }
                function onHeadlightButtonClicked(button)
                {
                    gIsHeadlightEnabled = !gIsHeadlightEnabled;
                    updateButtonEnabledText(button, gIsHeadlightEnabled);
                    isHeadlightEnabled = gIsHeadlightEnabled
                    postHttpRequest("setHeadlightEnabled", {isHeadlightEnabled})
                }

                
                function onCameraColorButtonClicked(button)
                {
                    gIsCameraColorEnabled = !gIsCameraColorEnabled;
                    updateButtonEnabledText(button, gIsCameraColorEnabled);
                    isCameraColorEnabled = gIsCameraColorEnabled
                    postHttpRequest("setCameraColorEnabled", {isCameraColorEnabled})
                
                }

                updateButtonEnabledText(document.getElementById("headlightId"), gIsHeadlightEnabled);
                updateDebugAnnotationButtonEnabledText(document.getElementById("debugAnnotationsId"), gAreDebugAnnotationsEnabled);
                updateButtonEnabledText(document.getElementById("cameraColorId"), gIsCameraColorEnabled);
                
                function updateIsSavePoseEnabled(updatedValue){
                gIsSavePoseEnabled = updatedValue
                }
                
                
                function handleDropDownSelect(selectObject)
                {
                    selectedIndex = selectObject.selectedIndex
                    itemName = selectObject.name
                    postHttpRequest("dropDownSelect", {selectedIndex, itemName});
                }

                function handleKeyActivity (e, actionType)
                {
                    var keyCode  = (e.keyCode ? e.keyCode : e.which);
                    var hasShift = (e.shiftKey ? 1 : 0)
                    var hasCtrl  = (e.ctrlKey  ? 1 : 0)
                    var hasAlt   = (e.altKey   ? 1 : 0)

                    if (actionType=="keyup")
                    {
                        if (keyCode == 76) // 'L'
                        {
                            // Simulate a click of the headlight button
                            onHeadlightButtonClicked(document.getElementById("headlightId"))
                        }
                        else if (keyCode == 79) // 'O'
                        {
                            // Simulate a click of the debug annotations button
                            onDebugAnnotationsButtonClicked(document.getElementById("debugAnnotationsId"))
                        }
                    }

                    postHttpRequest(actionType, {keyCode, hasShift, hasCtrl, hasAlt})
                }


                function handleTextInput(textField)
                {
                    textEntered = textField.value
                    postHttpRequest("sayText", {textEntered})
                }

                document.addEventListener("keydown", function(e) { handleKeyActivity(e, "keydown") } );
                document.addEventListener("keyup",   function(e) { handleKeyActivity(e, "keyup") } );
                
                function stopEventPropagation(event)
                {
                    if (event.stopPropagation)
                    {
                        event.stopPropagation();
                    }
                    else
                    {
                        event.cancelBubble = true
                    }
                }
                
                function saveImageClicked(textField)
                {
                fileName = textField.value
                postHttpRequest("saveImage", {fileName})
                }   
                
                function saveGainSettingsClicked(textField1)
                {
                gain = textField1.value
                postHttpRequest("adjustCameraGain", {gain})
                }
         
                function saveExposureSettingsClicked(textField1)
                {
                exposure = textField1.value
                postHttpRequest("adjustCameraExposure", {exposure})
                }


                document.getElementById("sayTextId").addEventListener("keydown", function(event) {
                    stopEventPropagation(event);
                } );
                document.getElementById("sayTextId").addEventListener("keyup", function(event) {
                    stopEventPropagation(event);
                } );
            </script>

        </body>
    </html>
    '''

def get_annotated_image():
    image = remote_control_cozmo.cozmo.world.latest_image
    if _display_debug_annotations != DEBUG_ANNOTATIONS_DISABLED:
        image = image.annotate_image(scale=2)
    else:
        image = image.raw_image
    return image

def streaming_video(url_root):
    '''Video streaming generator function'''
    try:
        while True:
            if remote_control_cozmo:
                image = get_annotated_image()

                img_io = io.BytesIO()
                image.save(img_io, 'PNG')
                img_io.seek(0)
                yield (b'--frame\r\n'
                       b'Content-Type: image/png\r\n\r\n' + img_io.getvalue() + b'\r\n')
            else:
                asyncio.sleep(.1)
    except cozmo.exceptions.SDKShutdown:
        # Tell the main flask thread to shutdown
        requests.post(url_root + 'shutdown')

def serve_single_image():
    if remote_control_cozmo:
        try:
            image = get_annotated_image()
            if image:
                return flask_helpers.serve_pil_image(image)
        except cozmo.exceptions.SDKShutdown:
            requests.post('shutdown')
    return flask_helpers.serve_pil_image(_default_camera_image)

def is_microsoft_browser(request):
    agent = request.user_agent.string
    return 'Edge/' in agent or 'MSIE ' in agent or 'Trident/' in agent


@flask_app.route("/cozmoImage")
def handle_cozmoImage():
    if is_microsoft_browser(request):
        return serve_single_image()
    return flask_helpers.stream_video(streaming_video, request.url_root)

def handle_key_event(key_request, is_key_down):
    message = json.loads(key_request.data.decode("utf-8"))
    global _is_save_pose_btn_enabled
    if remote_control_cozmo and _is_save_pose_btn_enabled:
        remote_control_cozmo.handle_key(key_code=(message['keyCode']), is_shift_down=message['hasShift'],
                                        is_ctrl_down=message['hasCtrl'], is_alt_down=message['hasAlt'],
                                        is_key_down=is_key_down)
    return ""

@flask_app.route('/shutdown', methods=['POST'])
def shutdown():
    flask_helpers.shutdown_flask(request)
    return ""

@flask_app.route('/isSavePoseBtnEnabled', methods=['POST'])
def getIsRobotPoseEnabled():
    global _is_save_pose_btn_enabled
    return to_js_bool_string(_is_save_pose_btn_enabled)

@flask_app.route('/toggleSavePoseBtn', methods=['POST'])
def toggleRobotPoseEnabled():
    remote_control_cozmo.disable_savepose_button()
    return to_js_bool_string(_is_save_pose_btn_enabled)

@flask_app.route('/saveRobotPose', methods=['POST'])
def handle_saveRobotPose():
    pose = remote_control_cozmo.get_robot_pose()
    return {'Pose': str(pose)}


@flask_app.route('/setHeadlightEnabled', methods=['POST'])
def handle_setHeadlightEnabled():
    '''Called from Javascript whenever headlight is toggled on/off'''
    message = json.loads(request.data.decode("utf-8"))
    if remote_control_cozmo:
        remote_control_cozmo.cozmo.set_head_light(enable=message['isHeadlightEnabled'])
    return ""


@flask_app.route('/setAreDebugAnnotationsEnabled', methods=['POST'])
def handle_setAreDebugAnnotationsEnabled():
    '''Called from Javascript whenever debug-annotations mode is toggled'''
    message = json.loads(request.data.decode("utf-8"))
    global _display_debug_annotations
    _display_debug_annotations = message['areDebugAnnotationsEnabled']
    if remote_control_cozmo:
        if _display_debug_annotations == DEBUG_ANNOTATIONS_ENABLED_ALL:
            remote_control_cozmo.cozmo.world.image_annotator.enable_annotator('robotState')
        else:
            remote_control_cozmo.cozmo.world.image_annotator.disable_annotator('robotState')
    return ""



@flask_app.route('/setCameraColorEnabled', methods=['POST'])
def handle_setCameraColorEnabled():
    message = json.loads(request.data.decode("utf-8"))
    if remote_control_cozmo:
        isCameraColorEnabled = message['isCameraColorEnabled']
        if isCameraColorEnabled:
            remote_control_cozmo.cozmo.camera.color_image_enabled = True
        else:
            remote_control_cozmo.cozmo.camera.color_image_enabled = False
    return ""


@flask_app.route('/keydown', methods=['POST'])
def handle_keydown():
    '''Called from Javascript whenever a key is down (note: can generate repeat calls if held down)'''
    return handle_key_event(request, is_key_down=True)


@flask_app.route('/keyup', methods=['POST'])
def handle_keyup():
    '''Called from Javascript whenever a key is released'''
    return handle_key_event(request, is_key_down=False)


@flask_app.route('/sayText', methods=['POST'])
def handle_sayText():
    '''Called from Javascript whenever the saytext text field is modified'''
    message = json.loads(request.data.decode("utf-8"))
    if remote_control_cozmo:
        remote_control_cozmo.text_to_say = message['textEntered']
    return ""


@flask_app.route('/updateCozmo', methods=['POST'])
def handle_updateCozmo():
    if remote_control_cozmo:
        remote_control_cozmo.update()
        action_queue_text = ""
        i = 1
        for action in remote_control_cozmo.action_queue:
            action_queue_text += str(i) + ": " + remote_control_cozmo.action_to_text(action) + "<br>"
            i += 1

        return '''Action Queue:<br>''' + action_queue_text + '''
        '''
    return ""

@flask_app.route('/saveImage', methods=['POST'])
def handle_save_image():
    message = json.loads(request.data.decode("utf-8"))
    file_name = message['fileName']
    if remote_control_cozmo:
        latest_image = remote_control_cozmo.cozmo.world.latest_image

        if latest_image is not None:
            # Scale the camera image down to fit on Cozmo's face
            # resized_image = latest_image.raw_image.resize(face_dimensions,
            #                                               Image.BICUBIC)
            imwrite_path = f"./remote_control_saved_images/{file_name}"
            # imwrite_path = f"./remote_control_saved_images/{date_timestamp}_{file_name}"
            if file_name == "" or file_name is None:
                date_timestamp = datetime.now().strftime('%m_%d_%H_%M_%S')
                imwrite_path = f"./remote_control_saved_images/{date_timestamp}.png"

            # img = preprocess_image(img_raw, (512, 512))

            # fit_size = (512, 512) # this is the size we use in yolo obj detection in retico
            # if fit_size == (latest_image.raw_image.width, latest_image.raw_image.height):
            #     image = latest_image.raw_image.copy()
            #     scale = 1
            # else:
            #     img_ratio = latest_image.raw_image.width / latest_image.raw_image.height
            #     fit_width, fit_height = fit_size
            #     fit_ratio = fit_width / fit_height
            #     if img_ratio > fit_ratio:
            #         fit_height = int(fit_width / img_ratio)
            #     elif img_ratio < fit_ratio:
            #         fit_width = int(fit_height * img_ratio)
            #     image = latest_image.raw_image.resize((fit_width, fit_height))
            image = latest_image.raw_image
            cv2.imwrite(imwrite_path, cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    return ""

@flask_app.route('/adjustCameraGain', methods=['POST'])
def adjust_camera_gain():
    message = json.loads(request.data.decode("utf-8"))
    gain_amount = float(message['gain'])
    if remote_control_cozmo:
        exposure_time = remote_control_cozmo.cozmo.camera.exposure_ms
        min_gain = remote_control_cozmo.cozmo.camera.config.min_gain
        max_gain = remote_control_cozmo.cozmo.camera.config.max_gain
        actual_gain = (1 - gain_amount) * min_gain + gain_amount * max_gain
        remote_control_cozmo.cozmo.camera.set_manual_exposure(exposure_time, actual_gain)
        print(f"Actual gain: {actual_gain}")
    return ""

@flask_app.route('/adjustCameraExposure', methods=['POST'])
def adjust_camera_exposure():
    message = json.loads(request.data.decode("utf-8"))
    exposure_amount = float(message['exposure'])
    if remote_control_cozmo:
        gain_amount = remote_control_cozmo.cozmo.camera.gain
        min_exposure = remote_control_cozmo.cozmo.camera.config.min_exposure_time_ms
        max_exposure = remote_control_cozmo.cozmo.camera.config.max_exposure_time_ms
        exposure_time = (1 - exposure_amount) * min_exposure + exposure_amount * max_exposure
        print(f"Exposure time: {exposure_time}")
        remote_control_cozmo.cozmo.camera.set_manual_exposure(exposure_time, gain_amount)
    return ""

