"""
Washer Cycle Detection AppDaemon App

This app monitors washer power consumption to accurately detect when a wash cycle
starts and ends, sending notifications accordingly.

Based on historical data analysis, the washer shows these patterns:
- Idle: 0.1-0.7 watts (standby power)
- Active phases: 10-350+ watts (washing, spinning, etc.)
- Pause periods: Can drop to idle levels between phases
- Cycle end: Returns to idle levels and stays there

The app uses trend analysis over multiple readings to avoid false positives
from brief pauses during the cycle.

NEW: Machine learning capabilities for improved predictions based on historical data.
"""

import appdaemon.plugins.hass.hassapi as hass
import time
import json
import os
from datetime import datetime, timedelta
from collections import deque
import logging
import statistics
from pathlib import Path

class WasherWatcher(hass.Hass):
    """
    Washer cycle detection app that monitors power consumption patterns
    """

    def initialize(self):
        """Initialize the app"""
        try:
            self.log("Initializing Washer Watcher")

            # Configuration validation
            self.persons = self.args.get("persons", [])
            if not isinstance(self.persons, list):
                self.log("Warning: 'persons' should be a list, using empty list", level="WARNING")
                self.persons = []

            self.washer_entity = self.args.get("washer_entity", "sensor.washer_power")
            if not self.washer_entity:
                self.log("Error: washer_entity is required", level="ERROR")
                return

            self.check_interval = self.args.get("check_interval", 30)  # seconds
            if self.check_interval < 5:
                self.log("Warning: check_interval too low, using minimum of 5 seconds", level="WARNING")
                self.check_interval = 5

            # Detection thresholds (based on historical data analysis)
            self.idle_threshold = self.args.get("idle_threshold", 1.0)  # watts
            self.active_threshold = self.args.get("active_threshold", 10.0)  # watts
            self.cycle_start_threshold = self.args.get("cycle_start_threshold", 15.0)  # watts
            self.trend_window = self.args.get("trend_window", 5)  # number of readings
            self.stable_idle_time = self.args.get("stable_idle_time", 300)  # seconds (5 minutes)

            # Strict detection parameters (ported from test_washer_script.py)
            self.base_idle_threshold = self.args.get("idle_threshold", 1.0)
            self.base_active_threshold = self.args.get("active_threshold", 10.0)
            self.cycle_start_threshold = self.args.get("cycle_start_threshold", 15.0)
            self.trend_window = self.args.get("trend_window", 10)
            self.stable_idle_time = self.args.get("stable_idle_time", 480)  # 8 minutes
            self.min_cycle_duration = self.args.get("min_cycle_duration", 180)  # 3 minutes
            self.max_power_threshold = self.args.get("max_power_threshold", 30.0)
            self.cooldown_time = self.args.get("cooldown_time", 300)  # 5 minutes cooldown
            self.power_spike_threshold = 20.0
            self.sustained_high_threshold = 12.0
            self.min_sustained_readings = 3
            self.adaptive_threshold_factor = 0.7
            self.recent_power_pattern = deque(maxlen=6)
            self.last_cycle_end_time = None
            self.cycle_validation_score = 0
            self.adaptive_idle_threshold = self.base_idle_threshold
            self.adaptive_active_threshold = self.base_active_threshold

            # Validate thresholds
            if self.idle_threshold >= self.active_threshold:
                self.log("Error: idle_threshold must be less than active_threshold", level="ERROR")
                return

            if self.active_threshold >= self.cycle_start_threshold:
                self.log("Error: active_threshold must be less than cycle_start_threshold", level="ERROR")
                return

            # Machine learning and historical data storage
            self.data_dir = self.args.get("data_dir", "washer_data")
            self.ensure_data_directory()

            # Load historical data
            self.historical_cycles = self.load_historical_data()

            # Cycle prediction data (will be updated with ML)
            self.cycle_predictions = self.calculate_predictions_from_history()

            # Current cycle tracking for ML
            self.current_cycle_data = {
                "start_time": None,
                "power_readings": [],
                "phases": [],
                "max_power": 0,
                "heating_detected": False,
                "high_spin_detected": False
            }

            # State tracking
            self.current_power = None
            self.previous_power = None
            self.power_history = deque(maxlen=self.trend_window)
            self.cycle_active = False
            self.cycle_start_time = None
            self.last_active_time = None
            self.stable_idle_start = None

            # Cycle prediction tracking (FIXED: Consistent variable names)
            self.cycle_type = None
            self.predicted_end_time = None
            self.heating_phase_detected = False  # FIXED: Consistent naming
            self.high_spin_phase_detected = False  # FIXED: Consistent naming

            # Notification tracking to prevent spam
            self.last_notification_time = {}
            self.notification_cooldown = 300  # 5 minutes between same type notifications

            # Initialize power history
            self.log(f"Washer entity: {self.washer_entity}")
            self.log(f"Monitoring {len(self.persons)} persons for notifications")
            self.log(f"Thresholds - Idle: {self.idle_threshold}W, Active: {self.active_threshold}W, Start: {self.cycle_start_threshold}W")
            self.log(f"Loaded {len(self.historical_cycles)} historical cycles for ML predictions")

            # Start monitoring
            self.run_every(self.check_power, "now", self.check_interval)

            # Start prediction monitoring (check every 5 minutes)
            self.run_every(self.check_prediction, "now", 300)

            # Listen for washer entity changes
            self.listen_state(self.on_power_change, self.washer_entity)

            self.log("Washer Watcher initialized successfully")

        except Exception as e:
            self.log(f"Error initializing Washer Watcher: {e}", level="ERROR")

    def ensure_data_directory(self):
        """Ensure the data directory exists"""
        try:
            Path(self.data_dir).mkdir(parents=True, exist_ok=True)
            self.log(f"Data directory ensured: {self.data_dir}")
        except Exception as e:
            self.log(f"Error creating data directory: {e}", level="ERROR")

    def load_historical_data(self):
        """Load historical cycle data from JSON file"""
        try:
            data_file = os.path.join(self.data_dir, "historical_cycles.json")
            if os.path.exists(data_file):
                with open(data_file, 'r') as f:
                    data = json.load(f)
                    self.log(f"Loaded {len(data)} historical cycles from {data_file}")
                    return data
            else:
                self.log("No historical data file found, starting fresh")
                return []
        except Exception as e:
            self.log(f"Error loading historical data: {e}", level="ERROR")
            return []

    def save_historical_data(self):
        """Save historical cycle data to JSON file"""
        try:
            data_file = os.path.join(self.data_dir, "historical_cycles.json")
            with open(data_file, 'w') as f:
                json.dump(self.historical_cycles, f, indent=2, default=str)
            self.log(f"Saved {len(self.historical_cycles)} historical cycles to {data_file}")
        except Exception as e:
            self.log(f"Error saving historical data: {e}", level="ERROR")

    def calculate_predictions_from_history(self):
        """Calculate predictions based on historical data"""
        try:
            if not self.historical_cycles:
                # Fallback to default predictions if no historical data
                return {
                    "long_heating": {
                        "average_duration": 544.5,  # minutes
                        "std_duration": 137.3,      # minutes
                        "min_duration": 179.0,      # minutes
                        "max_duration": 719.0,      # minutes
                        "heating_threshold": 2000,  # watts
                        "high_spin_threshold": 300  # watts
                    }
                }

            # Analyze historical cycles
            durations = []
            max_powers = []
            heating_cycles = []
            non_heating_cycles = []

            for cycle in self.historical_cycles:
                duration = cycle.get('duration_minutes', 0)
                max_power = cycle.get('max_power', 0)
                has_heating = cycle.get('heating_detected', False)

                if duration > 0:
                    durations.append(duration)
                    max_powers.append(max_power)

                    if has_heating:
                        heating_cycles.append(cycle)
                    else:
                        non_heating_cycles.append(cycle)

            # Calculate statistics
            predictions = {}

            if heating_cycles:
                heating_durations = [c.get('duration_minutes', 0) for c in heating_cycles if c.get('duration_minutes', 0) > 0]
                if heating_durations:
                    predictions["long_heating"] = {
                        "average_duration": statistics.mean(heating_durations),
                        "std_duration": statistics.stdev(heating_durations) if len(heating_durations) > 1 else 0,
                        "min_duration": min(heating_durations),
                        "max_duration": max(heating_durations),
                        "heating_threshold": 2000,  # watts
                        "high_spin_threshold": 300,  # watts
                        "cycle_count": len(heating_cycles)
                    }

            if non_heating_cycles:
                non_heating_durations = [c.get('duration_minutes', 0) for c in non_heating_cycles if c.get('duration_minutes', 0) > 0]
                if non_heating_durations:
                    predictions["quick_wash"] = {
                        "average_duration": statistics.mean(non_heating_durations),
                        "std_duration": statistics.stdev(non_heating_durations) if len(non_heating_durations) > 1 else 0,
                        "min_duration": min(non_heating_durations),
                        "max_duration": max(non_heating_durations),
                        "heating_threshold": 2000,  # watts
                        "high_spin_threshold": 300,  # watts
                        "cycle_count": len(non_heating_cycles)
                    }

            # Log prediction statistics
            for cycle_type, data in predictions.items():
                self.log(f"ML Predictions for {cycle_type}: {data['average_duration']:.1f}±{data['std_duration']:.1f} min (n={data['cycle_count']})")

            return predictions

        except Exception as e:
            self.log(f"Error calculating predictions from history: {e}", level="ERROR")
            # Return default predictions
            return {
                "long_heating": {
                    "average_duration": 544.5,
                    "std_duration": 137.3,
                    "min_duration": 179.0,
                    "max_duration": 719.0,
                    "heating_threshold": 2000,
                    "high_spin_threshold": 300
                }
            }

    def on_power_change(self, entity, attribute, old, new, kwargs):
        """Handle power reading changes"""
        try:
            if new is None or new == "unavailable":
                return

            power = float(new)

            # Validate power reading
            if power < 0:
                self.log(f"Warning: Negative power reading ignored: {power}W", level="WARNING")
                return
            if power > 10000:  # Unrealistic power reading
                self.log(f"Warning: Unrealistic power reading ignored: {power}W", level="WARNING")
                return

            self.process_power_reading(power)

        except (ValueError, TypeError) as e:
            self.log(f"Error processing power reading '{new}': {e}", level="ERROR")
        except Exception as e:
            self.log(f"Unexpected error in on_power_change: {e}", level="ERROR")

    def check_power(self, kwargs):
        """Periodic power check (fallback)"""
        try:
            power = self.get_state(self.washer_entity)
            if power is not None and power != "unavailable":
                try:
                    power_float = float(power)
                    self.process_power_reading(power_float)
                except (ValueError, TypeError) as e:
                    self.log(f"Error converting power reading '{power}' to float: {e}", level="ERROR")
        except Exception as e:
            self.log(f"Error in periodic power check: {e}", level="ERROR")

    def check_prediction(self, kwargs):
        """Periodic prediction check"""
        if self.cycle_active and self.predicted_end_time:
            self.send_prediction_notification()

    def update_adaptive_thresholds(self):
        if len(self.power_history) < 4:
            return
        idle_readings = [p for p in self.power_history if p < self.base_idle_threshold * 2]
        if idle_readings:
            avg_idle = sum(idle_readings) / len(idle_readings)
            self.adaptive_idle_threshold = max(0.3, avg_idle * 1.2)
        high_readings = [p for p in self.power_history if p > self.base_active_threshold]
        if high_readings:
            avg_high = sum(high_readings) / len(high_readings)
            self.adaptive_active_threshold = max(self.base_active_threshold, avg_high * 0.5)

    def detect_power_spike(self, power):
        if len(self.power_history) < 2:
            return False
        recent_avg = sum(list(self.power_history)[-2:]) / 2
        return power > recent_avg * 2.5 and power > self.power_spike_threshold

    def validate_cycle_start(self, power):
        self.cycle_validation_score = 0
        if self.detect_power_spike(power):
            self.cycle_validation_score += 2
        if len(self.recent_power_pattern) >= 4:
            recent_readings = list(self.recent_power_pattern)[-4:]
            high_count = sum(1 for p in recent_readings if p > self.sustained_high_threshold)
            if high_count >= 3:
                self.cycle_validation_score += 2
        if power > self.cycle_start_threshold:
            self.cycle_validation_score += 1
        if power > self.adaptive_idle_threshold * 2.5:
            self.cycle_validation_score += 1
        if len(self.power_history) >= 6:
            recent_readings = list(self.power_history)[-6:]
            high_readings = sum(1 for p in recent_readings if p > self.adaptive_active_threshold)
            if high_readings >= 4:
                self.cycle_validation_score += 1
        return self.cycle_validation_score >= 5

    def validate_cycle_end(self, power, timestamp=None):
        if not self.cycle_active:
            return False
        if power > self.max_power_threshold:
            return False
        if self.stable_idle_start is None:
            if timestamp:
                self.stable_idle_start = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            else:
                self.stable_idle_start = datetime.now()
            return False
        if timestamp:
            current_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        else:
            current_time = datetime.now()
        idle_duration = (current_time - self.stable_idle_start).total_seconds()
        if idle_duration < self.stable_idle_time:
            return False
        if self.last_active_time is None:
            return False
        time_since_active = (current_time - self.last_active_time).total_seconds()
        if time_since_active < self.stable_idle_time:
            return False
        if self.cycle_start_time:
            cycle_duration = (current_time - self.cycle_start_time).total_seconds()
            if cycle_duration < self.min_cycle_duration:
                return False
        if len(self.power_history) >= 6:
            recent_avg = sum(list(self.power_history)[-6:]) / 6
            if recent_avg > self.adaptive_idle_threshold * 1.5:
                return False
        return True

    def process_power_reading(self, power):
        """Process a new power reading and detect cycle changes"""
        try:
            self.previous_power = self.current_power
            self.current_power = power
            self.power_history.append(power)
            self.recent_power_pattern.append(power)
            self.update_adaptive_thresholds()

            # Update current cycle data for ML
            self.update_current_cycle_data(power)

            if len(self.power_history) < 4:
                return
            # Cooldown after cycle end
            if self.last_cycle_end_time:
                now = datetime.now()
                if (now - self.last_cycle_end_time).total_seconds() < self.cooldown_time:
                    return
            # Detect cycle start
            if not self.cycle_active and self.validate_cycle_start(power):
                self.cycle_started()
            # Update active time if running
            elif self.cycle_active and power > self.adaptive_active_threshold:
                self.last_active_time = datetime.now()
                if self.stable_idle_start:
                    self.stable_idle_start = None
                self.analyze_cycle_characteristics(power)
            # Detect cycle end
            elif self.cycle_active and self.validate_cycle_end(power):
                self.cycle_ended()
        except Exception as e:
            self.log(f"Error in process_power_reading: {e}", level="ERROR")

    def cycle_started(self):
        """Handle cycle start detection"""
        try:
            self.cycle_active = True
            self.cycle_start_time = datetime.now()
            self.last_active_time = datetime.now()
            self.stable_idle_start = None

            # Reset cycle prediction tracking
            self.cycle_type = None
            self.predicted_end_time = None
            self.heating_phase_detected = False
            self.high_spin_phase_detected = False

            # Initialize current cycle tracking for ML
            self.current_cycle_data = {
                "start_time": self.cycle_start_time.isoformat(),
                "power_readings": [],
                "phases": [],
                "max_power": 0,
                "heating_detected": False,
                "high_spin_detected": False
            }

            cycle_time = self.cycle_start_time.strftime("%H:%M")
            message = f"🧺 Washer cycle started at {cycle_time}"

            self.log(f"Cycle START detected at {cycle_time} (power: {self.current_power:.1f}W)")
            self.send_notifications(message, "washer_start")

        except Exception as e:
            self.log(f"Error in cycle_started: {e}", level="ERROR")

    def cycle_ended(self):
        """Handle cycle end detection"""
        try:
            if not self.cycle_active:
                return

            self.cycle_active = False
            end_time = datetime.now()
            cycle_time = end_time.strftime("%H:%M")

            # Calculate cycle duration
            duration = "unknown"
            duration_minutes = 0
            if self.cycle_start_time:
                duration_minutes = int((end_time - self.cycle_start_time).total_seconds() / 60)
                duration = f"{duration_minutes} minutes"

            # Store cycle data for ML
            self.store_cycle_data(duration_minutes)

            # Log cycle type and prediction accuracy if available
            prediction_info = ""
            if self.cycle_type and self.predicted_end_time and self.cycle_start_time:
                actual_duration = (end_time - self.cycle_start_time).total_seconds() / 60
                predicted_duration = self.cycle_predictions[self.cycle_type]["average_duration"]
                accuracy = abs(actual_duration - predicted_duration)
                prediction_info = f", cycle_type: {self.cycle_type}, prediction_accuracy: ±{accuracy:.1f}min"

            message = f"✅ Washer cycle finished at {cycle_time} (duration: {duration})"

            self.log(f"Cycle END detected at {cycle_time} (power: {self.current_power:.1f}W, duration: {duration}{prediction_info})")
            self.send_notifications(message, "washer_end")

            # Reset state (FIXED: Complete state reset)
            self.cycle_start_time = None
            self.last_active_time = None
            self.stable_idle_start = None
            self.cycle_type = None
            self.predicted_end_time = None
            self.heating_phase_detected = False
            self.high_spin_phase_detected = False
            self.current_cycle_data = {
                "start_time": None,
                "power_readings": [],
                "phases": [],
                "max_power": 0,
                "heating_detected": False,
                "high_spin_detected": False
            }
            self.last_cycle_end_time = datetime.now()

        except Exception as e:
            self.log(f"Error in cycle_ended: {e}", level="ERROR")

    def store_cycle_data(self, duration_minutes):
        """Store completed cycle data for machine learning"""
        try:
            if not self.current_cycle_data["start_time"]:
                return

            # Create cycle record
            cycle_record = {
                "start_time": self.current_cycle_data["start_time"],
                "end_time": datetime.now().isoformat(),
                "duration_minutes": duration_minutes,
                "max_power": self.current_cycle_data["max_power"],
                "heating_detected": self.current_cycle_data["heating_detected"],
                "high_spin_detected": self.current_cycle_data["high_spin_detected"],
                "power_readings_count": len(self.current_cycle_data["power_readings"]),
                "cycle_type": self.cycle_type or "unknown",
                "prediction_accuracy": None
            }

            # Calculate prediction accuracy if we had a prediction
            if self.cycle_type and self.predicted_end_time and self.cycle_start_time:
                actual_duration = duration_minutes
                predicted_duration = self.cycle_predictions[self.cycle_type]["average_duration"]
                accuracy = abs(actual_duration - predicted_duration)
                cycle_record["prediction_accuracy"] = accuracy

            # Add to historical data
            self.historical_cycles.append(cycle_record)

            # Keep only last 100 cycles to prevent file from growing too large
            if len(self.historical_cycles) > 100:
                self.historical_cycles = self.historical_cycles[-100:]
                self.log("Trimmed historical data to last 100 cycles")

            # Save to file
            self.save_historical_data()

            # Update predictions with new data
            self.cycle_predictions = self.calculate_predictions_from_history()

            self.log(f"Stored cycle data: {duration_minutes}min, max_power: {self.current_cycle_data['max_power']:.1f}W, heating: {self.current_cycle_data['heating_detected']}")

        except Exception as e:
            self.log(f"Error storing cycle data: {e}", level="ERROR")

    def update_current_cycle_data(self, power):
        """Update current cycle data for ML tracking"""
        try:
            if not self.cycle_active:
                return

            # Add power reading with timestamp
            reading = {
                "timestamp": datetime.now().isoformat(),
                "power": power
            }
            self.current_cycle_data["power_readings"].append(reading)

            # Update max power
            if power > self.current_cycle_data["max_power"]:
                self.current_cycle_data["max_power"] = power

            # Detect phases (FIXED: Use consistent thresholds)
            if power > 2000 and not self.current_cycle_data["heating_detected"]:
                self.current_cycle_data["heating_detected"] = True
                self.current_cycle_data["phases"].append({
                    "type": "heating",
                    "timestamp": datetime.now().isoformat(),
                    "power": power
                })

            if power > 300 and not self.current_cycle_data["high_spin_detected"]:
                self.current_cycle_data["high_spin_detected"] = True
                self.current_cycle_data["phases"].append({
                    "type": "high_spin",
                    "timestamp": datetime.now().isoformat(),
                    "power": power
                })

        except Exception as e:
            self.log(f"Error updating current cycle data: {e}", level="ERROR")

    def send_notifications(self, message, notification_type):
        """Send notifications to all configured persons with spam protection"""
        try:
            # Check notification cooldown
            now = datetime.now()
            if notification_type in self.last_notification_time:
                time_since_last = (now - self.last_notification_time[notification_type]).total_seconds()
                if time_since_last < self.notification_cooldown:
                    self.log(f"Notification {notification_type} skipped due to cooldown ({self.notification_cooldown - time_since_last:.0f}s remaining)")
                    return

            # Sanitize message
            sanitized_message = self._sanitize_message(message)
            title = self._sanitize_message(f"Washer Monitor - {notification_type.replace('_', ' ').title()}")

            # Send notifications
            success_count = 0
            for person in self.persons:
                try:
                    if not isinstance(person, dict):
                        self.log(f"Warning: Invalid person configuration: {person}", level="WARNING")
                        continue

                    person_name = person.get("name", "unknown")
                    notify_service = person.get("notify")

                    # Check if this notification type is enabled for this person
                    if not person.get(f"{notification_type}_enabled", True):
                        self.log(f"Skipping {notification_type} notification for {person_name} (disabled)")
                        continue

                    # Check if notify service is configured
                    if not notify_service:
                        self.log(f"No notify service configured for {person_name}", level="WARNING")
                        continue

                    # Use the correct AppDaemon notification format
                    self.call_service(
                        "notify/{}".format(notify_service),
                        title=title,
                        message=sanitized_message
                    )
                    self.log(f"Notification sent to {person_name}: {sanitized_message}")
                    success_count += 1

                except Exception as e:
                    self.log(f"Error sending notification to {person.get('name', 'unknown')}: {e}", level="ERROR")

            # Update cooldown if at least one notification was sent
            if success_count > 0:
                self.last_notification_time[notification_type] = now
                self.log(f"Sent {success_count} notification(s) for {notification_type}")
            else:
                self.log(f"No notifications sent for {notification_type}", level="WARNING")

        except Exception as e:
            self.log(f"Error in send_notifications: {e}", level="ERROR")

    def _sanitize_message(self, message):
        """Sanitize message for safe transmission"""
        try:
            # Remove any potentially problematic characters
            sanitized = str(message)
            # Replace any non-printable characters
            sanitized = ''.join(char for char in sanitized if char.isprintable() or char in ['\n', '\t'])
            # Limit length to prevent issues
            if len(sanitized) > 500:
                sanitized = sanitized[:497] + "..."
            return sanitized
        except Exception as e:
            self.log(f"Error sanitizing message: {e}", level="ERROR")
            return "Washer notification"

    def get_power_trend(self):
        """Get the current power trend"""
        if len(self.power_history) < 2:
            return "insufficient_data"

        recent = list(self.power_history)[-3:]
        older = list(self.power_history)[:-3] if len(self.power_history) > 3 else []

        if not older:
            return "stable"

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)

        if recent_avg > older_avg * 1.5:
            return "increasing"
        elif recent_avg < older_avg * 0.7:
            return "decreasing"
        else:
            return "stable"

    def log_status(self):
        """Log current status for debugging"""
        status = {
            "current_power": self.current_power,
            "cycle_active": self.cycle_active,
            "power_history": list(self.power_history),
            "trend": self.get_power_trend(),
            "stable_idle_start": self.stable_idle_start,
            "last_active_time": self.last_active_time,
            "cycle_type": self.cycle_type,
            "predicted_end_time": self.predicted_end_time,
            "heating_phase": self.heating_phase_detected,
            "high_spin_phase": self.high_spin_phase_detected
        }
        self.log(f"Status: {status}")

    def analyze_cycle_characteristics(self, power):
        """Analyze power patterns to determine cycle type and predict end time"""
        try:
            # Detect heating phase (2000+ watts)
            if power > self.cycle_predictions.get("long_heating", {}).get("heating_threshold", 2000):
                if not self.heating_phase_detected:
                    self.log(f"Heating phase detected (power: {power:.1f}W)")
                self.heating_phase_detected = True

            # Detect high spin phase (300+ watts)
            if power > self.cycle_predictions.get("long_heating", {}).get("high_spin_threshold", 300):
                if not self.high_spin_phase_detected:
                    self.log(f"High spin phase detected (power: {power:.1f}W)")
                self.high_spin_phase_detected = True

            # Determine cycle type based on characteristics and ML
            self.determine_cycle_type(power)

        except Exception as e:
            self.log(f"Error in analyze_cycle_characteristics: {e}", level="ERROR")

    def determine_cycle_type(self, power):
        """Determine cycle type using ML and current characteristics"""
        try:
            # If we already have a cycle type, don't change it
            if self.cycle_type:
                return

            # Use ML to predict cycle type based on current characteristics
            if self.heating_phase_detected:
                # Check if we have historical data for heating cycles
                if "long_heating" in self.cycle_predictions:
                    self.cycle_type = "long_heating"
                    self.log("Cycle type identified: long_heating (ML)")
                    self.update_prediction()
                else:
                    self.cycle_type = "long_heating"
                    self.log("Cycle type identified: long_heating (default)")
                    self.update_prediction()
            else:
                # Check if this looks like a quick wash based on power patterns
                if power < 1000 and len(self.power_history) >= 5:
                    # Analyze recent power pattern
                    recent_powers = list(self.power_history)[-5:]
                    avg_power = sum(recent_powers) / len(recent_powers)

                    if avg_power < 500 and "quick_wash" in self.cycle_predictions:
                        self.cycle_type = "quick_wash"
                        self.log("Cycle type identified: quick_wash (ML)")
                        self.update_prediction()

        except Exception as e:
            self.log(f"Error in determine_cycle_type: {e}", level="ERROR")

    def update_prediction(self):
        """Update cycle end prediction based on detected cycle type and ML data"""
        try:
            if not self.cycle_type or not self.cycle_start_time:
                return

            prediction_data = self.cycle_predictions.get(self.cycle_type)
            if not prediction_data:
                self.log(f"Warning: No prediction data for cycle type '{self.cycle_type}'", level="WARNING")
                return

            # Use ML-enhanced prediction
            predicted_duration_minutes = self.calculate_ml_prediction(prediction_data)
            predicted_duration_seconds = predicted_duration_minutes * 60

            # Calculate predicted end time
            self.predicted_end_time = self.cycle_start_time + timedelta(seconds=predicted_duration_seconds)

            # Log prediction with confidence
            predicted_time = self.predicted_end_time.strftime("%H:%M")
            confidence = self.calculate_prediction_confidence(prediction_data)
            self.log(f"ML Prediction: {self.cycle_type} cycle ending at {predicted_time} (±{prediction_data['std_duration']:.0f} min, confidence: {confidence:.1%})")

        except Exception as e:
            self.log(f"Error in update_prediction: {e}", level="ERROR")

    def calculate_ml_prediction(self, prediction_data):
        """Calculate ML-enhanced prediction duration"""
        try:
            base_duration = prediction_data["average_duration"]

            # If we have enough historical data, use weighted average
            if prediction_data.get("cycle_count", 0) >= 3:
                # Consider recent trends
                recent_cycles = [c for c in self.historical_cycles[-5:] if c.get("cycle_type") == self.cycle_type]
                if recent_cycles:
                    recent_avg = statistics.mean([c.get("duration_minutes", 0) for c in recent_cycles])
                    # Weight recent data more heavily
                    weighted_duration = (recent_avg * 0.7) + (base_duration * 0.3)
                    return weighted_duration

            return base_duration

        except Exception as e:
            self.log(f"Error in calculate_ml_prediction: {e}", level="ERROR")
            return prediction_data["average_duration"]

    def calculate_prediction_confidence(self, prediction_data):
        """Calculate confidence level for prediction"""
        try:
            cycle_count = prediction_data.get("cycle_count", 0)
            std_duration = prediction_data.get("std_duration", 0)
            avg_duration = prediction_data.get("average_duration", 0)

            if cycle_count == 0 or avg_duration == 0:
                return 0.5  # Default confidence

            # Confidence based on sample size and consistency
            sample_confidence = min(cycle_count / 10, 1.0)  # Max confidence at 10+ samples
            consistency_confidence = max(0.1, 1.0 - (std_duration / avg_duration))  # Lower std = higher confidence

            return (sample_confidence * 0.6) + (consistency_confidence * 0.4)

        except Exception as e:
            self.log(f"Error in calculate_prediction_confidence: {e}", level="ERROR")
            return 0.5

    def get_prediction_status(self):
        """Get current prediction status for notifications"""
        if not self.predicted_end_time:
            return None

        now = datetime.now()
        time_remaining = self.predicted_end_time - now

        if time_remaining.total_seconds() <= 0:
            return "overdue"
        elif time_remaining.total_seconds() <= 300:  # 5 minutes
            return "finishing_soon"
        else:
            minutes_remaining = int(time_remaining.total_seconds() / 60)
            return f"{minutes_remaining} minutes remaining"

    def send_prediction_notification(self):
        """Send prediction notification if cycle is taking longer than expected"""
        try:
            if not self.predicted_end_time:
                return

            now = datetime.now()
            time_remaining = self.predicted_end_time - now

            # Send notification if cycle is overdue or finishing soon
            if time_remaining.total_seconds() <= 0:
                overdue_minutes = abs(int(time_remaining.total_seconds() / 60))
                message = f"⚠️ Washer cycle is taking longer than expected (overdue by {overdue_minutes} minutes)"
                self.send_notifications(message, "washer_prediction")
            elif time_remaining.total_seconds() <= 300:  # 5 minutes
                remaining_minutes = int(time_remaining.total_seconds() / 60)
                message = f"⏰ Washer cycle finishing soon (predicted in {remaining_minutes} minutes)"
                self.send_notifications(message, "washer_prediction")

        except Exception as e:
            self.log(f"Error in send_prediction_notification: {e}", level="ERROR")

    def terminate(self):
        """Clean up when app is terminated"""
        try:
            self.log("Washer Watcher terminating - cleaning up")

            # Log final status if cycle is active
            if self.cycle_active:
                self.log(f"Warning: Cycle was active when terminating. Start time: {self.cycle_start_time}")

        except Exception as e:
            self.log(f"Error in terminate: {e}", level="ERROR")

    def get_status_summary(self):
        """Get a summary of current status for debugging"""
        try:
            status = {
                "cycle_active": self.cycle_active,
                "current_power": self.current_power,
                "power_history_length": len(self.power_history),
                "cycle_type": self.cycle_type,
                "predicted_end_time": self.predicted_end_time.isoformat() if self.predicted_end_time else None,
                "heating_phase_detected": self.heating_phase_detected,
                "high_spin_phase_detected": self.high_spin_phase_detected,
                "persons_configured": len(self.persons)
            }

            if self.cycle_active and self.cycle_start_time:
                elapsed = (datetime.now() - self.cycle_start_time).total_seconds() / 60
                status["elapsed_minutes"] = round(elapsed, 1)

            return status

        except Exception as e:
            self.log(f"Error in get_status_summary: {e}", level="ERROR")
            return {"error": str(e)}

    def test_against_historical_data(self, log_directory="data/washer"):
        """
        Test the script against historical log files to validate accuracy

        This function processes historical log files and compares the script's
        cycle detection with known cycle data to measure accuracy.
        """
        try:
            self.log("Starting historical data validation test...")

            # Statistics tracking
            total_files = 0
            processed_files = 0
            total_cycles_detected = 0
            total_cycles_expected = 0
            double_stops = 0
            missed_starts = 0
            missed_ends = 0
            false_starts = 0
            false_ends = 0

            # Timing accuracy tracking
            start_time_deltas = []
            end_time_deltas = []

            # Process each log file
            log_path = Path(log_directory)
            if not log_path.exists():
                self.log(f"Log directory not found: {log_directory}", level="ERROR")
                return

            log_files = list(log_path.glob("*.log"))
            total_files = len(log_files)
            self.log(f"Found {total_files} log files to process")

            for log_file in sorted(log_files):
                try:
                    self.log(f"Processing {log_file.name}...")
                    file_stats = self._process_single_log_file(log_file)

                    if file_stats:
                        processed_files += 1
                        total_cycles_detected += file_stats['cycles_detected']
                        total_cycles_expected += file_stats['cycles_expected']
                        double_stops += file_stats['double_stops']
                        missed_starts += file_stats['missed_starts']
                        missed_ends += file_stats['missed_ends']
                        false_starts += file_stats['false_starts']
                        false_ends += file_stats['false_ends']
                        start_time_deltas.extend(file_stats['start_deltas'])
                        end_time_deltas.extend(file_stats['end_deltas'])

                except Exception as e:
                    self.log(f"Error processing {log_file.name}: {e}", level="ERROR")

            # Calculate statistics
            self._generate_test_report(
                total_files, processed_files, total_cycles_detected, total_cycles_expected,
                double_stops, missed_starts, missed_ends, false_starts, false_ends,
                start_time_deltas, end_time_deltas
            )

        except Exception as e:
            self.log(f"Error in test_against_historical_data: {e}", level="ERROR")

    def _process_single_log_file(self, log_file):
        """Process a single log file and extract cycle information"""
        try:
            # Reset script state for testing
            self._reset_for_testing()

            # Parse log file to extract expected cycles
            expected_cycles = self._parse_log_file_for_cycles(log_file)

            # Simulate power readings from log file
            detected_cycles = self._simulate_power_readings(log_file)

            # Compare expected vs detected cycles
            comparison = self._compare_cycles(expected_cycles, detected_cycles)

            return comparison

        except Exception as e:
            self.log(f"Error processing log file {log_file}: {e}", level="ERROR")
            return None

    def _reset_for_testing(self):
        """Reset script state for testing"""
        self.current_power = None
        self.previous_power = None
        self.power_history.clear()
        self.cycle_active = False
        self.cycle_start_time = None
        self.last_active_time = None
        self.stable_idle_start = None
        self.cycle_type = None
        self.predicted_end_time = None
        self.heating_phase_detected = False
        self.high_spin_phase_detected = False
        self.current_cycle_data = {
            "start_time": None,
            "power_readings": [],
            "phases": [],
            "max_power": 0,
            "heating_detected": False,
            "high_spin_detected": False
        }

    def _parse_log_file_for_cycles(self, log_file):
        """Parse log file to extract expected cycle start/end times"""
        expected_cycles = []

        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Look for cycle start/end patterns in log
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Extract timestamp and look for cycle indicators
                # This is a simplified parser - adjust based on actual log format
                if "cycle" in line.lower() or "start" in line.lower() or "end" in line.lower():
                    # Try to extract timestamp and cycle info
                    cycle_info = self._extract_cycle_info_from_line(line)
                    if cycle_info:
                        expected_cycles.append(cycle_info)

            self.log(f"Extracted {len(expected_cycles)} expected cycles from {log_file.name}")
            return expected_cycles

        except Exception as e:
            self.log(f"Error parsing log file {log_file}: {e}", level="ERROR")
            return []

    def _extract_cycle_info_from_line(self, line):
        """Extract cycle information from a log line"""
        try:
            # Parse the actual log format: "2025-07-06 13:11:23 washer state: running power: 27.2 -> 21.4"
            import re

            # Extract timestamp and state information
            pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) washer state: (\w+) power:'
            match = re.search(pattern, line)

            if not match:
                return None

            timestamp = match.group(1)
            state = match.group(2).lower()

            # Determine if this is a start or end indicator
            if state == 'running':
                return {'type': 'start', 'timestamp': timestamp, 'state': state, 'line': line}
            elif state == 'idle':
                return {'type': 'end', 'timestamp': timestamp, 'state': state, 'line': line}

            return None

        except Exception as e:
            self.log(f"Error extracting cycle info from line: {e}", level="ERROR")
            return None

    def _simulate_power_readings(self, log_file):
        """Simulate power readings from log file to test detection"""
        detected_cycles = []

        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Extract power readings and timestamps
            power_readings = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Look for power readings (adjust pattern based on actual format)
                power_info = self._extract_power_reading_from_line(line)
                if power_info:
                    power_readings.append(power_info)

            # Sort by timestamp
            power_readings.sort(key=lambda x: x['timestamp'])

            # Simulate the detection algorithm
            for reading in power_readings:
                self.process_power_reading(reading['power'])

                # Check if cycle state changed
                if self.cycle_active and not hasattr(self, '_last_cycle_state'):
                    self._last_cycle_state = True
                    detected_cycles.append({
                        'type': 'start',
                        'timestamp': reading['timestamp'],
                        'power': reading['power']
                    })
                elif not self.cycle_active and hasattr(self, '_last_cycle_state') and self._last_cycle_state:
                    self._last_cycle_state = False
                    detected_cycles.append({
                        'type': 'end',
                        'timestamp': reading['timestamp'],
                        'power': reading['power']
                    })

            self.log(f"Detected {len(detected_cycles)} cycles from {log_file.name}")
            return detected_cycles

        except Exception as e:
            self.log(f"Error simulating power readings from {log_file}: {e}", level="ERROR")
            return []

    def _extract_power_reading_from_line(self, line):
        """Extract power reading from a log line"""
        try:
            # Parse the actual log format: "2025-07-06 13:11:23 washer state: running power: 27.2 -> 21.4"
            import re

            # Extract timestamp and power values
            pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) washer state: \w+ power: (?:[\d.]+|unavailable) -> ([\d.]+)'
            match = re.search(pattern, line)

            if not match:
                return None

            timestamp = match.group(1)
            power_str = match.group(2)

            # Skip if power is unavailable
            if power_str == 'unavailable':
                return None

            power = float(power_str)

            if power >= 0:
                return {'timestamp': timestamp, 'power': power}

            return None

        except Exception as e:
            self.log(f"Error extracting power reading from line: {e}", level="ERROR")
            return None

    def _compare_cycles(self, expected_cycles, detected_cycles):
        """Compare expected vs detected cycles and calculate accuracy metrics"""
        try:
            stats = {
                'cycles_expected': len(expected_cycles) // 2,  # Assuming start/end pairs
                'cycles_detected': len(detected_cycles) // 2,
                'double_stops': 0,
                'missed_starts': 0,
                'missed_ends': 0,
                'false_starts': 0,
                'false_ends': 0,
                'start_deltas': [],
                'end_deltas': []
            }

            # Group expected cycles by start/end pairs
            expected_pairs = []
            for i in range(0, len(expected_cycles), 2):
                if i + 1 < len(expected_cycles):
                    start_cycle = expected_cycles[i]
                    end_cycle = expected_cycles[i + 1]
                    if start_cycle['type'] == 'start' and end_cycle['type'] == 'end':
                        expected_pairs.append((start_cycle, end_cycle))

            # Group detected cycles by start/end pairs
            detected_pairs = []
            for i in range(0, len(detected_cycles), 2):
                if i + 1 < len(detected_cycles):
                    start_cycle = detected_cycles[i]
                    end_cycle = detected_cycles[i + 1]
                    if start_cycle['type'] == 'start' and end_cycle['type'] == 'end':
                        detected_pairs.append((start_cycle, end_cycle))

            # Compare pairs
            for exp_start, exp_end in expected_pairs:
                # Find matching detected cycle
                best_match = None
                best_start_delta = float('inf')
                best_end_delta = float('inf')

                for det_start, det_end in detected_pairs:
                    # Calculate time deltas
                    start_delta = self._calculate_time_delta(exp_start['timestamp'], det_start['timestamp'])
                    end_delta = self._calculate_time_delta(exp_end['timestamp'], det_end['timestamp'])

                    # Check if this is a reasonable match (within 10 minutes)
                    if abs(start_delta) <= 600 and abs(end_delta) <= 600:
                        if abs(start_delta) + abs(end_delta) < best_start_delta + best_end_delta:
                            best_match = (det_start, det_end)
                            best_start_delta = start_delta
                            best_end_delta = end_delta

                if best_match:
                    stats['start_deltas'].append(best_start_delta)
                    stats['end_deltas'].append(best_end_delta)
                else:
                    stats['missed_starts'] += 1
                    stats['missed_ends'] += 1

            # Count false positives
            for det_start, det_end in detected_pairs:
                # Check if this detected cycle matches any expected cycle
                matched = False
                for exp_start, exp_end in expected_pairs:
                    start_delta = self._calculate_time_delta(exp_start['timestamp'], det_start['timestamp'])
                    end_delta = self._calculate_time_delta(exp_end['timestamp'], det_end['timestamp'])
                    if abs(start_delta) <= 600 and abs(end_delta) <= 600:
                        matched = True
                        break

                if not matched:
                    stats['false_starts'] += 1
                    stats['false_ends'] += 1

            # Count double stops (multiple end detections for same cycle)
            # This would require more sophisticated analysis of the detection sequence

            return stats

        except Exception as e:
            self.log(f"Error comparing cycles: {e}", level="ERROR")
            return None

    def _calculate_time_delta(self, time1_str, time2_str):
        """Calculate time difference between two timestamp strings in seconds"""
        try:
            # Parse timestamps using the actual format: "2025-07-06 13:11:23"
            from datetime import datetime

            time1 = datetime.strptime(time1_str, '%Y-%m-%d %H:%M:%S')
            time2 = datetime.strptime(time2_str, '%Y-%m-%d %H:%M:%S')

            delta = (time2 - time1).total_seconds()
            return delta

        except Exception as e:
            self.log(f"Error calculating time delta: {e}", level="ERROR")
            return 0

    def _generate_test_report(self, total_files, processed_files, total_cycles_detected,
                            total_cycles_expected, double_stops, missed_starts, missed_ends,
                            false_starts, false_ends, start_time_deltas, end_time_deltas):
        """Generate comprehensive test report"""
        try:
            self.log("=" * 60)
            self.log("HISTORICAL DATA VALIDATION REPORT")
            self.log("=" * 60)

            # File processing stats
            self.log(f"Files processed: {processed_files}/{total_files}")
            self.log(f"Success rate: {(processed_files/total_files*100):.1f}%" if total_files > 0 else "N/A")

            # Cycle detection stats
            self.log(f"\nCycle Detection Statistics:")
            self.log(f"Expected cycles: {total_cycles_expected}")
            self.log(f"Detected cycles: {total_cycles_detected}")

            if total_cycles_expected > 0:
                detection_rate = (total_cycles_detected / total_cycles_expected) * 100
                self.log(f"Detection rate: {detection_rate:.1f}%")

            # Error analysis
            self.log(f"\nError Analysis:")
            self.log(f"Missed starts: {missed_starts}")
            self.log(f"Missed ends: {missed_ends}")
            self.log(f"False starts: {false_starts}")
            self.log(f"False ends: {false_ends}")
            self.log(f"Double stops: {double_stops}")

            # Timing accuracy
            if start_time_deltas:
                avg_start_delta = sum(start_time_deltas) / len(start_time_deltas)
                std_start_delta = statistics.stdev(start_time_deltas) if len(start_time_deltas) > 1 else 0
                self.log(f"\nStart Time Accuracy:")
                self.log(f"Average delta: {avg_start_delta:.1f} seconds ({avg_start_delta/60:.1f} minutes)")
                self.log(f"Standard deviation: {std_start_delta:.1f} seconds ({std_start_delta/60:.1f} minutes)")
                self.log(f"Min delta: {min(start_time_deltas):.1f} seconds ({min(start_time_deltas)/60:.1f} minutes)")
                self.log(f"Max delta: {max(start_time_deltas):.1f} seconds ({max(start_time_deltas)/60:.1f} minutes)")

                # Count positive/negative deltas
                positive_deltas = [d for d in start_time_deltas if d > 0]
                negative_deltas = [d for d in start_time_deltas if d < 0]
                self.log(f"Later detections: {len(positive_deltas)} (avg: {sum(positive_deltas)/len(positive_deltas)/60:.1f} min)" if positive_deltas else "Later detections: 0")
                self.log(f"Earlier detections: {len(negative_deltas)} (avg: {sum(negative_deltas)/len(negative_deltas)/60:.1f} min)" if negative_deltas else "Earlier detections: 0")

            if end_time_deltas:
                avg_end_delta = sum(end_time_deltas) / len(end_time_deltas)
                std_end_delta = statistics.stdev(end_time_deltas) if len(end_time_deltas) > 1 else 0
                self.log(f"\nEnd Time Accuracy:")
                self.log(f"Average delta: {avg_end_delta:.1f} seconds ({avg_end_delta/60:.1f} minutes)")
                self.log(f"Standard deviation: {std_end_delta:.1f} seconds ({std_end_delta/60:.1f} minutes)")
                self.log(f"Min delta: {min(end_time_deltas):.1f} seconds ({min(end_time_deltas)/60:.1f} minutes)")
                self.log(f"Max delta: {max(end_time_deltas):.1f} seconds ({max(end_time_deltas)/60:.1f} minutes)")

                # Count positive/negative deltas
                positive_deltas = [d for d in end_time_deltas if d > 0]
                negative_deltas = [d for d in end_time_deltas if d < 0]
                self.log(f"Later detections: {len(positive_deltas)} (avg: {sum(positive_deltas)/len(positive_deltas)/60:.1f} min)" if positive_deltas else "Later detections: 0")
                self.log(f"Earlier detections: {len(negative_deltas)} (avg: {sum(negative_deltas)/len(negative_deltas)/60:.1f} min)" if negative_deltas else "Earlier detections: 0")

            # Overall accuracy score
            if total_cycles_expected > 0:
                accuracy_score = ((total_cycles_detected - false_starts - false_ends) / total_cycles_expected) * 100
                self.log(f"\nOverall Accuracy Score: {accuracy_score:.1f}%")

            self.log("=" * 60)

        except Exception as e:
            self.log(f"Error generating test report: {e}", level="ERROR")