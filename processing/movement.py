

def calculate_speed(coordinate_vector_curr, coordinate_vector_prev, frame_offset, frame_duration_s, frame_number):
    difference_vector = (coordinate_vector_curr - coordinate_vector_prev)
    print("______________________")
    print("Frame: ", frame_number, ", velocity calculate for last ", (frame_offset * frame_duration_s), " s")
    print("Velocity Vector: ", -difference_vector / (frame_offset * frame_duration_s))
    print("______________________")
