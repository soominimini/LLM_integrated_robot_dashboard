# Robot Movement During Speech

This feature adds gentle arm and head movements to the QTrobot while it speaks, making the interaction more natural and engaging.

## Features

### 🤖 Automatic Movement
- **Head Movements**: Gentle yaw and pitch movements while speaking
- **Arm Movements**: Subtle shoulder and elbow movements for both arms
- **Synchronized**: Movements are timed with speech duration
- **Non-blocking**: Movements run in separate threads to avoid blocking speech

### 🎛️ User Control
- **Enable/Disable**: Toggle movement on/off from the dashboard
- **Real-time Status**: See current movement status
- **Visual Feedback**: Movement indicator during story reading

## How It Works

### Movement Types

1. **Head Movements**:
   - Single movement at the beginning of each sentence
   - Larger random yaw movements (±8 degrees)
   - Larger random pitch movements (±5 degrees)
   - No continuous movement during speech
   - Smooth transition to new position

2. **Arm Movements**:
   - Single movement at the beginning of each sentence
   - Larger shoulder pitch movements (±6 degrees)
   - Larger shoulder roll movements (±4 degrees)
   - Larger elbow roll movements (±4 degrees)
   - Independent movement for left and right arms
   - No continuous movement during speech

### Technical Implementation

- **Kinematic Interface**: Uses the existing `QTrobotKinematicInterface` for precise joint control
- **Threading**: Movement runs in separate threads to avoid blocking speech
- **Duration Estimation**: Speech duration estimated at 0.1 seconds per character
- **Safety**: Movements are small and gentle to avoid discomfort

## Usage

### Dashboard Control

1. **Access Dashboard**: Log in to see the main dashboard
2. **Movement Settings**: Find the "🤖 Robot Movement Settings" section
3. **Toggle Switch**: Use the toggle to enable/disable movement
4. **Status Display**: See current status (Enabled/Disabled/Not Available)

### Story Reading

1. **Read Stories**: Navigate to any story reading interface
2. **Speak Sentences**: Click "🔊 Speak Sentence" button
3. **Movement Indicator**: Watch for the green movement indicator
4. **Visual Feedback**: See "🤖 Robot is moving while speaking..." message

## API Endpoints

### Movement Settings
```http
POST /api/movement_settings
Content-Type: application/json

{
  "enabled": true
}
```

### Movement Status
```http
GET /api/movement_status
```

Response:
```json
{
  "success": true,
  "movement_available": true,
  "movement_enabled": true
}
```

## Configuration

### Movement Parameters

The movement behavior can be customized by modifying these parameters in `src/tts_helper.py`:

```python
# Head movement ranges (degrees)
yaw_offset = random.uniform(-8, 8)      # Left/right movement (increased)
pitch_offset = random.uniform(-5, 5)    # Up/down movement (increased)

# Arm movement ranges (degrees)
right_offset = [random.uniform(-6, 6), random.uniform(-4, 4), random.uniform(-4, 4)]
left_offset = [random.uniform(-6, 6), random.uniform(-4, 4), random.uniform(-4, 4)]

# Movement timing (seconds)
head_interval = random.uniform(0.5, 1.5)
arm_interval = random.uniform(0.8, 2.0)
```

### Safety Limits

- **Head Yaw**: ±8 degrees maximum (increased from ±4)
- **Head Pitch**: ±5 degrees maximum (increased from ±2)
- **Arm Shoulder Pitch**: ±6 degrees maximum (increased from ±3)
- **Arm Shoulder Roll**: ±4 degrees maximum (increased from ±2)
- **Arm Elbow Roll**: ±4 degrees maximum (increased from ±2)
- **Movement Timing**: Single movement at sentence start

## Testing

Run the test scripts to verify movement functionality:

```bash
# Test general movement
python test_movement.py

# Test specific head movement improvements
python test_head_movement.py
```

This will:
1. Test TTS availability
2. Test movement availability
3. Test joint limits and safe ranges
4. Test current head position monitoring
5. Test speech with movement enabled
6. Test speech with movement disabled

The head movement test specifically:
1. Shows current head position
2. Displays joint limits and safe ranges
3. Tests improved pitch movement behavior
4. Checks final head position for drift

## Troubleshooting

### Movement Not Available
- Check if ROS is running
- Verify kinematic interface is properly initialized
- Check robot hardware connections

### Movement Too Fast/Slow
- Adjust timing parameters in the code
- Modify movement ranges for gentler motion
- Check robot joint limits

### Movement Interferes with Speech
- Movement runs in separate threads
- Speech should not be affected
- Check for resource conflicts

## Dependencies

- `kinematics.kinematic_interface.QTrobotKinematicInterface`
- ROS (Robot Operating System)
- `qt_robot_interface` services
- Threading support

## Safety Notes

- Movements are designed to be gentle and non-threatening
- All movements are within safe joint limits
- Movement can be disabled at any time
- Emergency stop functionality is preserved

## Future Enhancements

- **Emotion-based Movement**: Different movement patterns for different emotions
- **Speech-synchronized**: Movements timed with speech patterns
- **User Preferences**: Individual movement style preferences
- **Learning**: Adaptive movement based on user interaction 