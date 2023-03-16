from plane_grasp_real import PlaneGraspClass
import time
if __name__ == '__main__':
    g = PlaneGraspClass(
        saved_model_path='trained-models/jacquard-rgbd-grconvnet3-drop0-ch32/epoch_48_iou_0.93',
        visualize=True,
        include_rgb=True
    )
    grasp_result =[]
    iter=0
    while True:
        grasp_success=g.generate()
        if grasp_success:
            grasp_result.append(True)
        else:
            grasp_result.append(False)
        # end
        if (iter>=2) and (not grasp_result[iter]) and (not grasp_result[iter-1]) and (not grasp_result[iter-2]):
            print('grasp_result_array:',grasp_result)
            break
        iter += 1
        time.sleep(0.2)
