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
            hour_angle = minute_angle

        return hour_angle, minute_angle

    #get both angles and convert them into an actual time
    def angles_to_time(self, hour_angle, minute_angle):

        #380dg/60min = 1min is 6 degrees, %60 to stay between o and 59
        minute = round(minute_angle / 6) % 60

        #360dg/12 = 1h is 30 degrees
        hour = hour_angle / 30

        #adjust the hour depending how many minutes have passed
        hour = hour - minute / 60
        hour = round(hour) % 12

        #edge case for the clock angle being 0
        if hour == 0:
            hour = 12

        return hour, minute
