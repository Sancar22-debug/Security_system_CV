# Surveillance Intrusion Detection System

This is a video surveillance system that can find people who walk into forbidden areas.

## What You Can Do With It

- Draw a forbidden zone on any video frame
- The system will watch the video and check if anyone enters your zone
- If someone enters, it shows a red "ALARM" warning
- The alarm stops 3 seconds after the person leaves the zone

## How to Start

First, install what you need:
```
pip install -r requirements.txt
```

Then make your forbidden zone:
```
python zone.py
```
Click on the video to draw corners of your zone. Press S to save it, Q to quit.

Now run the detection:
```
python main.py
```
Press Q to stop the program.

## How It Works Inside

The system uses YOLOv8 - a smart AI that can see people in video. It gives each person an ID number and follows them from frame to frame.
When YOLO finds a person, the system checks if their feet are inside your forbidden zone. It only alarms after seeing the person inside for 5 frames in a row - this stops false alarms from mistakes.
If a person leaves the video and comes back, the system tries to remember them using color information (HSV histogram) and position. This stops ID numbers from changing too much (but it still changes too much ;-;).

## Files in This Project

- **main.py** - The main program that watches video and finds intruders
- **zone.py** - A small tool to draw forbidden zones on video
- **restricted_zones.json** - Where your zone corners are saved (Delete this if you want to try it on another video)
- **requirements.txt** - List of software packages you need
- **video/test.mp4** - The test video you should use (or change it to another video with same naming to use)

## Important Settings

In main.py you can change these numbers:
- `confidence_treshold`: How sure YOLO must be (0.45 means 45% sure)
- `cooldown`: How many seconds before alarm stops (3.0)
- `required_frames`: How many frames of intrusion needed for alarm (5)
- `cache_ttl`: How long to remember lost people (3.0 seconds)

## About This Work

This is one of my starting computer vision project. I built something that little bit works, I know the code is not perfect. Some parts are more complex than they should be, and I repeated some code that I could have written once.
I did this because I wanted to reduce false alarms and make the system reliable.

**I want to learn from real developers.** I hope to see how professional systems are built, understand best practices, and help in any way I can while I learn.
