def calculate_zone_value(ElementCount, GoldCount, RockCount, ZoneScalar):
    return ZoneScalar * ((4 * GoldCount) + (4 * ElementCount) - RockCount)

def find_zones(mine):
    n = len(mine)
    ZoneScalar = 1  # You can adjust this based on your preference

    max_value = float('-inf')
    max_zones = []

    for left in range(n - 4):
        # Initialize counts for the current zone
        ElementCount = {'g': 0, 'o': 0, 'l': 0, 'd': 0}
        GoldCount = 0
        RockCount = 0

        for right in range(left, n):
            # Update counts based on the current character
            if mine[right] in ElementCount:
                ElementCount[mine[right]] += 1
                GoldCount += 1
            elif mine[right] == '-':
                RockCount += 1

            # Check if the zone is valid (non-overlapping)
            if right - left >= 4:
                # Calculate the zone value
                value = calculate_zone_value(ElementCount['d'], GoldCount, RockCount, ZoneScalar)

                # Update max_value and max_zones
                if value > max_value:
                    max_value = value
                    max_zones = [(left, right)]

    # Print the left and right indexes of the five zones
    for zone in max_zones[:5]:
        print(zone[0], zone[1])

