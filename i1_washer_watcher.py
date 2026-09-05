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
"""

import appdaemon.plugins.hass.hassapi as hass
import json
import os
import time
from datetime import datetime, timedelta
from collections import deque
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
            trend_window = self.args.get("trend_window", 10)  # number of readings
            # The validation scoring hardcodes checks over the last 6 readings,
            # so a window below 6 silently deletes those criteria -- with 5,
            # the washer could never reach the start-detection bar at all
            # (T-05). Floored rather than refused: the deployed washer config
            # says 5, and a raise here would take a working notification path
            # down over one number, while the floor restores the behaviour the
            # config author plainly wanted.
            self.trend_window = max(6, int(trend_window))
            if self.trend_window != trend_window:
                self.log(
                    f"trend_window {trend_window} is below the 6 readings the "
                    f"validation scoring inspects; using {self.trend_window}",
                    level="WARNING")
            self.stable_idle_time = self.args.get("stable_idle_time", 480)  # seconds (8 minutes)
            self.min_cycle_duration = self.args.get("min_cycle_duration", 180)  # 3 minutes minimum
            self.max_power_threshold = self.args.get("max_power_threshold", 30.0)  # watts
            self.cooldown_time = self.args.get("cooldown_time", 300)  # 5 minutes cooldown
            self.power_spike_threshold = 20.0  # watts
            self.sustained_high_threshold = 12.0  # watts

            # State tracking
            self.current_power = None
            self.power_history = deque(maxlen=self.trend_window)
            self.recent_power_pattern = deque(maxlen=6)
            self.cycle_active = False
            self.cycle_start_time = None
            self.last_active_time = None
            self.stable_idle_start = None
            self.last_cycle_end_time = None
            self.cycle_validation_score = 0
            self.adaptive_idle_threshold = self.idle_threshold
            self.adaptive_active_threshold = self.active_threshold

            # Android companion-app delivery settings. The default HA notification channel
            # can be disabled on the phone, which silently discards every notification sent
            # to it - HA reports success and nothing arrives. Sending on a dedicated channel
            # keeps these alerts independent of that setting and lets them be muted on
            # their own without affecting other apps. See backlog T-52.
            self.notification_channel = self.args.get("notification_channel", "appliance_alerts")
            self.notification_priority = self.args.get("notification_priority", "high")

            # Validate thresholds
            if self.idle_threshold >= self.active_threshold:
                self.log("Error: idle_threshold must be less than active_threshold", level="ERROR")
                return

            if self.active_threshold >= self.cycle_start_threshold:
                self.log("Error: active_threshold must be less than cycle_start_threshold", level="ERROR")
                return

            # Machine learning and historical data storage
            self.data_dir = self.args.get("data_dir", ".")

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

            # Cycle prediction tracking
            self.cycle_type = None
            self.predicted_end_time = None
            self.heating_phase_detected = False
            self.high_spin_phase_detected = False

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
            data_file = os.path.join(self.data_dir, "washer_historical_cycles.json")
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
            data_file = os.path.join(self.data_dir, "washer_historical_cycles.json")
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
        """Periodic prediction check. Logs only -- deliberately does not notify.

        This used to call send_prediction_notification(), which produced three
        messages for a single cycle with the countdown running backwards
        ("finishing soon (predicted in -13 minutes)"). Owner's decision
        2026-09-05: notify when the cycle starts and when it actually ends,
        nothing in between. See T-58.
        """
        if self.cycle_active and self.predicted_end_time:
            self.log_prediction_status()

    def update_adaptive_thresholds(self):
        """Update adaptive thresholds based on recent power readings"""
        if len(self.power_history) < 4:
            return
        idle_readings = [p for p in self.power_history if p < self.idle_threshold * 2]
        if idle_readings:
            avg_idle = sum(idle_readings) / len(idle_readings)
            self.adaptive_idle_threshold = max(0.3, avg_idle * 1.2)
        high_readings = [p for p in self.power_history if p > self.active_threshold]
        if high_readings:
            avg_high = sum(high_readings) / len(high_readings)
            self.adaptive_active_threshold = max(self.active_threshold, avg_high * 0.5)

    def detect_power_spike(self, power):
        """Detect power spikes -- against the readings BEFORE this one.

        The reading is appended to power_history before validation runs, so the
        old comparison against history[-2:] included `power` itself:
        power > 2.5*(prev+power)/2 simplifies to -0.5*power > 2.5*prev, which
        no positive wattage can satisfy. The points this criterion carries were
        unreachable for the app's whole life (T-05). For the washer (criteria
        2/2/1/1/1, bar 5) that capped the maximum score at 4 -- it could never
        detect a cycle start. The disher's deliberately looser scoring
        (3/3/2/2/2, bar 6) still cleared the bar without the spike, which is
        why it worked; the dead criterion cost it sensitivity, not function.
        """
        prior = list(self.power_history)[:-1]
        if len(prior) < 2:
            return False
        recent_avg = sum(prior[-2:]) / 2
        return power > recent_avg * 2.5 and power > self.power_spike_threshold

    def validate_cycle_start(self, power):
        """Validate cycle start with scoring system"""
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

    def validate_cycle_end(self, power):
        """Validate cycle end"""
        if not self.cycle_active:
            return False
        if power > self.max_power_threshold:
            return False
        if self.stable_idle_start is None:
            self.stable_idle_start = time.time()
            return False
        idle_duration = time.time() - self.stable_idle_start
        if idle_duration < self.stable_idle_time:
            return False
        if self.last_active_time is None:
            return False
        time_since_active = time.time() - self.last_active_time
        if time_since_active < self.stable_idle_time:
            return False
        if self.cycle_start_time:
            cycle_duration = time.time() - self.cycle_start_time
            if cycle_duration < self.min_cycle_duration:
                return False
        if len(self.power_history) >= 6:
            recent_avg = sum(list(self.power_history)[-6:]) / 6
            if recent_avg > self.adaptive_idle_threshold * 1.5:
                return False
        return True

    def process_power_reading(self, power):
        """Process a power reading and detect cycle changes"""
        try:
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
                now = time.time()
                if now - self.last_cycle_end_time < self.cooldown_time:
                    return

            # Detect cycle start
            if not self.cycle_active and self.validate_cycle_start(power):
                self.cycle_started()

            # Update active time if running
            elif self.cycle_active and power > self.adaptive_active_threshold:
                self.last_active_time = time.time()
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
            self.cycle_start_time = time.time()
            self.last_active_time = time.time()
            self.stable_idle_start = None

            # Reset cycle prediction tracking
            self.cycle_type = None
            self.predicted_end_time = None
            self.heating_phase_detected = False
            self.high_spin_phase_detected = False

            # Initialize current cycle tracking for ML
            self.current_cycle_data = {
                "start_time": self._wall(self.cycle_start_time).isoformat(),
                "power_readings": [],
                "phases": [],
                "max_power": 0,
                "heating_detected": False,
                "high_spin_detected": False
            }

            cycle_time = self._wall(self.cycle_start_time).strftime("%H:%M")
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
            end_time = time.time()
            cycle_time = self._wall(end_time).strftime("%H:%M")

            # Calculate cycle duration
            duration = "unknown"
            duration_minutes = 0
            if self.cycle_start_time:
                duration_minutes = int((end_time - self.cycle_start_time) / 60)
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

            # Reset state
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
            self.last_cycle_end_time = time.time()

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
                "end_time": self.get_now().isoformat(),
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
                "timestamp": self.get_now().isoformat(),
                "power": power
            }
            self.current_cycle_data["power_readings"].append(reading)

            # Update max power
            if power > self.current_cycle_data["max_power"]:
                self.current_cycle_data["max_power"] = power

            # Detect phases
            if power > 2000 and not self.current_cycle_data["heating_detected"]:
                self.current_cycle_data["heating_detected"] = True
                self.current_cycle_data["phases"].append({
                    "type": "heating",
                    "timestamp": self.get_now().isoformat(),
                    "power": power
                })

            if power > 300 and not self.current_cycle_data["high_spin_detected"]:
                self.current_cycle_data["high_spin_detected"] = True
                self.current_cycle_data["phases"].append({
                    "type": "high_spin",
                    "timestamp": self.get_now().isoformat(),
                    "power": power
                })

        except Exception as e:
            self.log(f"Error updating current cycle data: {e}", level="ERROR")

    def _wall(self, epoch: float) -> datetime:
        """The wall-clock reading of an epoch stamp, in Home Assistant's zone.

        All timing state in this app is epoch floats (S8-02, T-07): durations
        survive the DST fold that naive-datetime subtraction lies across.
        Only display goes through here, and fromtimestamp(tz=...) is
        fold-correct where get_now() + timedelta arithmetic would not be.
        """
        return datetime.fromtimestamp(epoch, tz=self.get_now().tzinfo)

    def send_notifications(self, message, notification_type):
        """Send notifications to all configured persons with spam protection"""
        try:
            # Check notification cooldown
            now = time.time()
            if notification_type in self.last_notification_time:
                time_since_last = now - self.last_notification_time[notification_type]
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
                        message=sanitized_message,
                        data=self._notification_data()
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

    def _notification_data(self) -> dict:
        """Build the companion-app data block for a notification.

        Returns the Android delivery hints every notify call in this app must carry:
        a dedicated channel, plus priority/ttl so the message is not deferred by Doze.
        Returns an empty dict if no channel is configured, so the caller can pass it
        unconditionally.
        """
        if not self.notification_channel:
            return {}
        data = {"channel": self.notification_channel}
        if self.notification_priority:
            data["priority"] = self.notification_priority
            data["ttl"] = 0
        return data

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
            self.predicted_end_time = self.cycle_start_time + predicted_duration_seconds

            # Log prediction with confidence
            predicted_time = self._wall(self.predicted_end_time).strftime("%H:%M")
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

    def log_prediction_status(self):
        """Log how the cycle is tracking against its prediction. Never notifies.

        Replaces send_prediction_notification (T-58). That method branched on
        `time_remaining <= -900` for "overdue" and `<= 300` for "finishing
        soon", and because neither branch had a said-this-already latch, the
        second stayed true from the predicted end until 15 minutes past it --
        firing every 5 minutes with an ever more negative countdown -- and then
        handed over to the first, which repeated indefinitely until the cycle
        genuinely ended. The per-type cooldown paced that rather than stopping
        it.

        The estimate itself is worth keeping: it is how the ML durations get
        tuned. It is the notification that was wrong, not the number. Note the
        clamp at zero -- a "remaining" time is never negative; past the
        prediction the honest word is "overdue".
        """
        try:
            if not self.predicted_end_time:
                return

            remaining = self.predicted_end_time - time.time()
            if remaining >= 0:
                self.log(
                    f"Washer cycle tracking: ~{int(remaining / 60)} min "
                    f"remaining (predicted end "
                    f"{self._wall(self.predicted_end_time).strftime('%H:%M')})",
                    level="DEBUG",
                )
            else:
                self.log(
                    f"Washer cycle is {int(-remaining / 60)} min past its "
                    f"predicted end of "
                    f"{self._wall(self.predicted_end_time).strftime('%H:%M')} "
                    f"and still running -- not notifying, the end event will",
                    level="DEBUG",
                )

        except Exception as e:
            self.log(f"Error in log_prediction_status: {e}", level="ERROR")

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
                "predicted_end_time": self._wall(self.predicted_end_time).isoformat() if self.predicted_end_time else None,
                "heating_phase_detected": self.heating_phase_detected,
                "high_spin_phase_detected": self.high_spin_phase_detected,
                "persons_configured": len(self.persons)
            }

            if self.cycle_active and self.cycle_start_time:
                elapsed = (time.time() - self.cycle_start_time) / 60
                status["elapsed_minutes"] = round(elapsed, 1)

            return status

        except Exception as e:
            self.log(f"Error in get_status_summary: {e}", level="ERROR")
            return {"error": str(e)}