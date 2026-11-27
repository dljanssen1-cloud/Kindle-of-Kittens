import math

class TimeReader:

    # line become length and angle
    def line_info(self, line, center_x, center_y):
        x1, y1, x2, y2 = line  #coordinates of a line (x,y)(x,y) start end point

        #distance from both endpoints to the clock center
        distance_1 = math.hypot(x1 - center_x, y1 - center_y)
        distance_2 = math.hypot(x2 - center_x, y2 - center_y)

        #defining choice for deciding which is the tip
        if distance_1 > distance_2:
            tip_x = x1
            tip_y = y1
            length = distance_1
        else:
            tip_x = x2
            tip_y = y2
            length = distance_2

        #get the direction of the hand
        direction_x = tip_x - center_x
        direction_y = tip_y - center_y

        angle = math.degrees(math.atan2(direction_x, -direction_y))
        angle = angle % 360 #stay between 0 to 359

        return angle, length

    #returns smallest difference between two angles
    def angle_difference(self, angle_1, angle_2):
        difference = abs(angle_1 - angle_2)

        #for angles that cross 360 degrees
        if difference > 180:
            difference = 360 - difference

        return difference

    #chooses which the hour and minute hands
    def pick_hands(self, lines, center_x, center_y):
        if len(lines) == 0:
            return None

        hands = []

        #use line_info on all lines
        for line in lines:
            angle, length = self.line_info(line, center_x, center_y)
            hands.append((angle, length))

        #sort hands from longest to shortest
        hands.sort(key=lambda hand: hand[1], reverse=True)

        #longest line is the minute hand
        minute_angle = hands[0][0]
        hour_angle = None

        #find the second longest line that points in a different direction
        for angle, length in hands[1:]:
            difference = self.angle_difference(angle, minute_angle)

            if difference > 15:
                hour_angle = angle
                break

        #if both hands point in the same direction
        if hour_angle is None:
            return None

        return hour_angle, minute_angle

    def pick_polar_hands(self, peaks, polar, radius, hand_detector):

        if len(peaks) < 2:
            return None

        # Only consider the strongest polar candidates
        candidates = peaks[:6]

        best_hands = None
        best_score = float("-inf")

        max_score = candidates[0][1]

        if max_score <= 0:
            return None

        # Estimate the length of each candidate
        hands = []

        for angle, score in candidates:

            length = hand_detector.estimate_polar_length(
                polar, angle, radius
            )

            hands.append((angle, score, length))

        # Test different combinations of hour and minute hands
        for hour_angle, hour_score, hour_length in hands:

            for minute_angle, minute_score, minute_length in hands:

                # Do not use the same candidate twice
                if hour_angle == minute_angle:
                    continue

                # Check mathematical time consistency
                error = self.time_consistency(
                    hour_angle, minute_angle
                )

                # Normalise the contrast scores
                contrast_score = (
                    hour_score + minute_score
                ) / max_score

                # Reward combinations with a longer minute hand
                length_score = (
                    minute_length - hour_length
                ) / radius

                # Combine all three features
                score = contrast_score - error / 30

                # Keep the best combination
                if score > best_score:

                    best_score = score

                    best_hands = (
                        hour_angle,
                        minute_angle
                    )

        return best_hands

    #get both angles and convert them into an actual time
    def angles_to_time(self, hour_angle, minute_angle):

        # Calculate the unrounded minutes
        exact_minute = minute_angle / 6

        # Calculate the hour while accounting for minute movement
        hour = round(
            (hour_angle - 0.5 * exact_minute) / 30
        ) % 12

        # Round the minutes
        minute = round(exact_minute)

        # Handle rounding into the next hour
        if minute == 60:
            minute = 0
            hour = (hour + 1) % 12

        if hour == 0:
            hour = 12

        return hour, minute

    def time_consistency(self, hour_angle, minute_angle):

        # Estimate the minutes
        minute = minute_angle / 6

        # Estimate the hour
        hour = round(
            (hour_angle - 0.5 * minute) / 15
        ) % 12

        # Calculate the expected hour hand angle
        expected_hour_angle = (
            hour * 30 + minute * 0.5
        ) % 360

        # Calculate the angular error
        error = self.angle_difference(
            hour_angle,
            expected_hour_angle
        )

        return error
