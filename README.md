# Washer & Dishwasher Cycle Detection System

Copyright (c) 2025 the_louie

This AppDaemon app monitors washer and dishwasher power consumption to accurately detect when wash cycles start and end, sending notifications accordingly.

## How It Works

The system analyzes power consumption patterns from your washer and dishwasher to distinguish between:
- **Idle state**: 0.1-1.2 watts (standby power)
- **Active phases**: 10-350+ watts (washing, spinning, etc.)
- **Heating phases**: 2000+ watts (drying, sanitizing)
- **Pause periods**: Brief drops to idle levels between phases
- **Cycle completion**: Sustained return to idle levels

### Key Features

1. **Trend Analysis**: Uses multiple power readings to detect sustained changes, not just spikes
2. **False Positive Prevention**: Waits for stable idle periods before declaring cycle end
3. **Double Stop Protection**: Handles false stops in dishwasher cycles (unique to dishwashers)
4. **Real-time Monitoring**: Responds to power changes immediately via state listeners
5. **Fallback Polling**: Periodic checks ensure no events are missed
6. **Configurable Thresholds**: Adjustable power levels and timing parameters
7. **Cycle Prediction**: Predicts cycle end times based on historical data analysis
8. **Smart Notifications**: Sends notifications for cycle start, end, and predictions

## Installation

1. Copy `i1_washer_watcher.py` and `i1_disher_watcher.py` to your AppDaemon `apps` directory
2. Copy the configuration from `config.yaml` to your AppDaemon configuration
3. Restart AppDaemon

## Configuration

### Basic Configuration

```yaml
# Washer Configuration
washer_watcher:
  module: i1_washer_watcher
  class: WasherWatcher
  washer_entity: sensor.washer_power
  persons:
    - name: louie
      notify: mobile_app_iphone_28

# Dishwasher Configuration
disher_watcher:
  module: i1_disher_watcher
  class: DisherWatcher
  disher_entity: sensor.disher_power
  persons:
    - name: louie
      notify: mobile_app_iphone_28
```

### Advanced Configuration

```yaml
washer_watcher:
  module: i1_washer_watcher
  class: WasherWatcher

  # Power sensor entity
  washer_entity: sensor.washer_power

  # Detection thresholds (watts)
  idle_threshold: 1.0          # Power level considered idle
  active_threshold: 10.0       # Power level considered active
  cycle_start_threshold: 15.0  # Power level to trigger start detection

  # Timing parameters
  check_interval: 30           # Check frequency (seconds)
  trend_window: 5              # Readings for trend analysis
  stable_idle_time: 300        # Idle time before cycle end (seconds)

  # Notifications
  persons:
    - name: louie
      notify: mobile_app_iphone_28
    - name: spouse
      notify: notify.telegram

disher_watcher:
  module: i1_disher_watcher
  class: DisherWatcher

  # Power sensor entity
  disher_entity: sensor.disher_power

  # Detection thresholds (watts) - LOOSER SENSITIVITY
  idle_threshold: 1.2          # Power level considered idle (lower to catch more cycles)
  active_threshold: 12.0       # Power level considered active (lower to catch more cycles)
  cycle_start_threshold: 15.0  # Power level to trigger start detection (lower to catch more cycles)

  # Timing parameters
  check_interval: 30           # Check frequency (seconds)
  trend_window: 10             # Readings for trend analysis
  stable_idle_time: 600        # Idle time before cycle end (10 minutes - shorter for disher)
  min_cycle_duration: 480      # Minimum cycle duration (8 minutes - shorter for disher)
  cooldown_time: 600           # Cooldown between cycles (10 minutes - shorter for disher)

  # Notifications
  persons:
    - name: louie
      notify: mobile_app_iphone_28
    - name: spouse
      notify: notify.telegram
```

## Detection Logic

### Washer Cycle Detection

#### Cycle Start Detection

A washer cycle is considered started when:
1. Power exceeds `cycle_start_threshold` (default: 15W)
2. Power is significantly higher than idle (3x `idle_threshold`)
3. Power remains above `active_threshold` for 3 consecutive readings

#### Cycle End Detection

A washer cycle is considered ended when:
1. Power drops to `idle_threshold` or below
2. Stays idle for `stable_idle_time` (default: 5 minutes)
3. No active periods detected during the idle time
4. Had a previous active cycle (prevents false starts)

### Dishwasher Cycle Detection

#### Cycle Start Detection

A dishwasher cycle is considered started when:
1. Power exceeds `cycle_start_threshold` (default: 15W) - **LOOSER THRESHOLD**
2. Power spike detection (2.5x recent average)
3. Sustained high readings (3+ readings above 12W)
4. Validation score of 6+ points (reduced from 8)

#### Cycle End Detection

A dishwasher cycle is considered ended when:
1. Power drops to `idle_threshold` or below
2. Stays idle for `stable_idle_time` (default: 10 minutes) - **SHORTER TIME**
3. **Double stop protection**: Ignores false stops within 10 minutes
4. Minimum cycle duration met (8 minutes minimum)
5. No recent high power readings

#### Double Stop Protection

Dishwashers have a unique issue where they can appear to stop mid-cycle:
- **False stops**: Brief drops to idle during heating phases
- **Protection**: System waits 10 minutes before considering a real stop
- **Multiple false stops**: Can handle up to 3 false stops per cycle
- **Resume detection**: Automatically resumes cycle tracking if activity continues

## Historical Data Analysis

Based on analysis of your appliance logs, the system handles:

### Washer Characteristics
- **Power ranges**: 0.1W (idle) to 2000+W (heating/spinning)
- **Pause periods**: Brief drops to idle between phases
- **Variable cycle lengths**: 30 minutes to several hours
- **Multiple phases**: Washing, rinsing, spinning, etc.

### Dishwasher Characteristics
- **Power ranges**: 0.1W (idle) to 2000+W (heating/drying)
- **Lower overall consumption**: 10-60W during washing phases
- **Heating phases**: 2000+W during drying/sanitizing
- **False stops**: Brief pauses between heating phases
- **Longer cycles**: 2-6 hours typical duration

### Cycle Types Identified

#### Washer Cycle Types

**Long Heating Cycle** (544.5 ± 137.3 minutes):
- **Duration**: 179-719 minutes (3-12 hours)
- **Characteristics**: Includes heating phase (2000+ watts) and high spin phase (300+ watts)
- **Typical use**: Heavy loads, hot water cycles, sanitize cycles
- **Prediction accuracy**: ±137 minutes (2+ hours)

#### Dishwasher Cycle Types

**Standard Wash Cycle** (120 minutes typical):
- **Duration**: 60-180 minutes (1-3 hours)
- **Characteristics**: Includes heating phase (2000+ watts) and high spin phase (100+ watts)
- **Typical use**: Normal dishwashing with drying
- **Prediction accuracy**: Based on historical data

**Quick Wash Cycle** (60 minutes typical):
- **Duration**: 30-90 minutes (0.5-1.5 hours)
- **Characteristics**: Lower power consumption, no heating phase
- **Typical use**: Light loads, quick cleaning
- **Prediction accuracy**: Based on historical data

## Troubleshooting

### False Start Notifications

If you get false start notifications:
- **Washer**: Increase `cycle_start_threshold` and `active_threshold`
- **Dishwasher**: Increase `cycle_start_threshold` (but keep lower than washer)
- Check for other appliances on the same circuit

### False End Notifications

If you get false end notifications:
- **Washer**: Increase `stable_idle_time`, decrease `idle_threshold`
- **Dishwasher**: Increase `stable_idle_time`, check for false stops
- Check if appliance has long pause periods

### No Notifications

If you don't get notifications:
- Verify `washer_entity`/`disher_entity` points to correct sensor
- Check notification service configuration
- Review AppDaemon logs for errors

### Dishwasher-Specific Issues

**Missing cycles**:
- The dishwasher uses looser thresholds to catch more cycles
- If still missing cycles, decrease thresholds further
- Check for very brief cycles that might be below minimum duration

**False stops detected as real ends**:
- The double stop protection should handle this automatically
- If issues persist, increase `false_stop_cooldown` time

## Logging

The app logs all detection events and errors. Check your AppDaemon logs for:
- Cycle start/end detection
- Power readings and trends
- False stop detection (dishwasher only)
- Cycle type identification and predictions
- Notification delivery status
- Configuration errors

## Prediction Features

### Cycle End Prediction

The system predicts cycle end times based on historical data analysis:

1. **Automatic Detection**: Identifies cycle type when heating phase is detected
2. **Real-time Updates**: Continuously monitors power patterns during cycle
3. **Smart Notifications**: Sends alerts when:
   - Cycle is finishing soon (within 5 minutes)
   - Cycle is overdue (taking longer than predicted)
4. **Accuracy**: Predictions improve with more historical data

### Notification Types

#### Washer Notifications
- **🧺 Cycle Start**: When a new washer cycle begins
- **✅ Cycle End**: When washer cycle completes successfully
- **⏰ Finishing Soon**: When cycle is predicted to end within 5 minutes
- **⚠️ Overdue**: When cycle is taking longer than expected

#### Dishwasher Notifications
- **🍽️ Cycle Start**: When a new dishwasher cycle begins
- **✅ Cycle End**: When dishwasher cycle completes successfully
- **⏰ Finishing Soon**: When cycle is predicted to end within 5 minutes
- **⚠️ Overdue**: When cycle is taking longer than expected

## Data Structure

The system stores historical cycle data in JSON format for machine learning predictions:

**Washer Data**: `washer_historical_cycles.json`
**Dishwasher Data**: `disher_historical_cycles.json`

Each cycle record contains:
- Start/end times
- Duration in minutes
- Maximum power consumption
- Detected phases (heating, high spin)
- Cycle type classification
- Prediction accuracy metrics

The system processes power readings in this format:
```
YYYY-MM-DD HH:MM:SS [device] state: [state] power: [old_value] -> [new_value]
```

Example:
```
2025-07-06 13:11:23 washer state: running power: 27.2 -> 21.4
2025-07-06 14:30:15 disher state: running power: 15.8 -> 12.3
```

## Files

- `i1_washer_watcher.py` - Main washer AppDaemon app
- `i1_disher_watcher.py` - Main dishwasher AppDaemon app
- `config.yaml` - Configuration example for both appliances
- `washer_historical_cycles.json` - Historical washer cycle data (ML)
- `disher_historical_cycles.json` - Historical dishwasher cycle data (ML)

## License

This project is licensed under the BSD 2-Clause License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 the_louie