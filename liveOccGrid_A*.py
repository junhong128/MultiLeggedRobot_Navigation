########################################################
#real-time occupancy grid construction & A* navigation
#selectively scans a single row 
#builds path using A*
########################################################



import cv2
import pyrealsense2 as rs
import numpy as np
from astar_pathfinding import PathPlanner

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

##############################################   PARAMS   ##############################################
#grid params
cell = 0.04
left, right = -0.35, 0.35
depthMax = 0.75

#display params
scale = 20
robotRadiusCell = 0



##############################################   FUNCTIONS   ##############################################
def pickGoal(grid: np.ndarray) -> tuple[int, int] | None:
    rows, cols = grid.shape
    topRow = 0
    centerCol = cols // 2

    #check if cell occupied
    if grid[topRow, centerCol] == 0:
        return (topRow, centerCol)

    #find closest free cell
    for dx in range(1, cols):
        for c in (centerCol - dx, centerCol + dx):
            if 0 <= c < cols and grid[topRow, c] == 0:
                return (topRow, c)

    return None
def drawPath(imgBGR: np.ndarray, path: list[tuple[int,int]], scale: int):
    if not path or len(path) < 2:
        return
    pts = np.array([[c*scale + scale//2, r*scale + scale//2] for (r, c) in path], dtype=np.int32)
    cv2.polylines(imgBGR, [pts], isClosed=False, thickness=2, color=(255, 0, 0))  # blue-ish
    for p in pts:
        cv2.circle(imgBGR, p, radius=2, color=(255, 0, 0), thickness=-1)



try:
    #get center of cam
    colorStream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = colorStream.get_intrinsics()
    ppx, ppy, fx, fy = intr.ppx, intr.ppy, intr.fx, intr.fy
    cy = int(round(intr.ppy))

    #get meters per depth unit
    depthSensor = profile.get_device().first_depth_sensor()
    depthScale = depthSensor.get_depth_scale()

    #cells
    sideCellNo = int(np.ceil((right - left) / cell))
    depthCellNo = int(np.ceil(depthMax / cell))

    #planner init
    planner = None
    start = None
    goal = None

    while True:
        frames = align.process(pipeline.wait_for_frames())
        depthFrame = frames.get_depth_frame()
        colorFrame = frames.get_color_frame()
        if not depthFrame or not colorFrame:
            continue
        
        color = np.asanyarray(colorFrame.get_data())
        depth = np.asanyarray(depthFrame.get_data())
        
        #get 1 row on cam
        h, w = depth.shape
        row = max(0, min(h - 1, cy))
        depthRow = depth[row:row+1, :]
        colorRow = color[row:row+1, :]

        #convert to meter depths
        depthRowM = (depthRow.astype(np.float32) * depthScale)[0]

        #grid (0-free, 1-occupied)
        grid = np.zeros((depthCellNo, sideCellNo), dtype=np.uint8)

        for i in range(w):
            z = float(depthRowM[i])
            if 0 < z < depthMax:
                x = float((i - ppx) / fx * z)
                colIndex = int((x - left) / cell)
                rowIndex = int(z / cell)
                if 0 <= colIndex < sideCellNo and 0 <= rowIndex < depthCellNo:
                    grid[rowIndex, colIndex] = 1

        #flip top row to bottom row
        grid = np.flipud(grid)

        #planner init
        if planner is None:
            planner = PathPlanner(grid, robot_radius=robotRadiusCell)
            rows, cols = grid.shape
            start = (rows - 1, cols // 2)
            goal = pickGoal(grid)
            if goal is not None:
                raw = planner.find_path(start, goal)
                if raw:
                    planner.current_path = planner.smooth_path(raw)
                    planner.current_path_index = 0
        else:
            currPosition = start if planner.current_path is None else planner.current_path[min(planner.current_path_index, len(planner.current_path)-1)]
            planner.dynamic_replan(currPosition, goal, new_occupancy_grid=grid, replan_threshold=5)



        ##############################################   DISPLAY   ##############################################
        #grid (white-0-free, black-1-occupied)
        gridImg = (1 - grid) * 255
        gridImg = cv2.resize(gridImg, (gridImg.shape[1]*scale, gridImg.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
        vis = cv2.cvtColor(gridImg, cv2.COLOR_GRAY2BGR)
        
        #start n goal
        if start:
            cv2.circle(vis, (start[1]*scale + scale//2, start[0]*scale + scale//2), 5, (0, 255, 0), -1)
        if goal:
            cv2.circle(vis, (goal[1]*scale + scale//2, goal[0]*scale + scale//2), 5, (0, 0, 255), -1)

        #path
        if planner and planner.current_path:
            drawPath(vis, planner.current_path, scale)

        cv2.imshow("LiveOccGrid + A*", vis)



        #change goal
        if cv2.waitKey(1) == ord('r'):
            goal = pickGoal(grid)
            planner.current_path = None
            planner.current_path_index = 0

        #close
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
