def get_anchor(coords):
    """Returns the top left, and bottom right anchor of the shape"""
    min_x, min_y = 10000, 10000
    max_x, max_y = 0, 0
    for x, y in coords:
        min_x = min(x, min_x)
        min_y = min(y, min_y)
        max_x = max(x, max_x)
        max_y = max(y, max_y)
    return (min_x, min_y), (max_x, max_y)


def get_pixel_coords(grid):
    """Gets the coords of all the pixel values"""
    pixel_coord = {}
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            for value in "abcdefghij":
                if grid[row][col] == value:
                    if value in pixel_coord:
                        pixel_coord[value].append((row, col))
                    else:
                        pixel_coord[value] = [(row, col)]
    return dict(sorted(pixel_coord.items(), key=lambda x: -len(x[1])))


def create_object(grid, coords):
    """Create an object based on the existing grid, and the coordinates of it"""
    (min_x, min_y), (max_x, max_y) = get_anchor(coords)
    newgrid = [
        ["." for _ in range(max_y - min_y + 1)] for _ in range(max_x - min_x + 1)
    ]
    for x, y in coords:
        if grid[x][y] == ".":
            newgrid[x - min_x][y - min_y] = "$"
        else:
            newgrid[x - min_x][y - min_y] = grid[x][y]
    return {"tl": (min_x, min_y), "grid": newgrid}


def get_objects(
    grid,
    diag=False,
    multicolor=False,
    by_row=False,
    by_col=False,
    by_color=False,
    more_info=True,
):
    rows = len(grid)
    cols = len(grid[0])
    visited = set()
    objects = []
    missing_color = False

    # check whether there is a missing color
    for each in grid:
        for cell in each:
            if cell == "j":
                missing_color = True

    def is_valid(grid, row, col, value):
        # multicolor can return any cell as long as it is not visited and not a blank
        if multicolor:
            return (
                0 <= row < rows
                and 0 <= col < cols
                and (row, col) not in visited
                and grid[row][col] != "."
                and grid[row][col] != "$"
            )
        else:
            return (
                0 <= row < rows
                and 0 <= col < cols
                and (row, col) not in visited
                and grid[row][col] == value
            )

    def dfs(grid, row, col, value):
        if is_valid(grid, row, col, value):
            visited.add((row, col))
            object_coords.add((row, col))

            if not by_row:
                dfs(grid, row - 1, col, value)  # up
                dfs(grid, row + 1, col, value)  # down
            if not by_col:
                dfs(grid, row, col - 1, value)  # left
                dfs(grid, row, col + 1, value)  # right
            if not by_row and not by_col and diag:
                dfs(grid, row - 1, col - 1, value)  # top-left diagonal
                dfs(grid, row - 1, col + 1, value)  # top-right diagonal
                dfs(grid, row + 1, col - 1, value)  # bottom-left diagonal
                dfs(grid, row + 1, col + 1, value)  # bottom-right diagonal

    # # if by color, we don't need to do dfs
    if by_color:
        pixels = get_pixel_coords(grid)
        for key, value in pixels.items():
            object_coords = value
            object_dict = create_object(grid, object_coords)
            if more_info:
                object_dict["size"] = (
                    len(object_dict["grid"]),
                    len(object_dict["grid"][0]),
                )
                # object_dict['br']=(object_dict['tl'][0]+object_dict['size'][0]-1,object_dict['tl'][1]+object_dict['size'][1]-1)
                object_dict["cell_count"] = len(object_coords)
                # object_dict['shape'] = generate_hash([['x' if cell != '.' else '.' for cell in row] for row in object_dict['grid']])
                object_dict["shape"] = [
                    ["x" if cell != "." else "." for cell in row]
                    for row in object_dict["grid"]
                ]
            objects.append(object_dict)

    else:
        for row in range(rows):
            for col in range(cols):
                value = grid[row][col]
                if (row, col) not in visited:
                    if value == "." or value == 0:
                        continue
                    object_coords = set()
                    dfs(grid, row, col, value)
                    object_dict = create_object(grid, object_coords)
                    if more_info:
                        object_dict["size"] = (
                            len(object_dict["grid"]),
                            len(object_dict["grid"][0]),
                        )
                        object_dict["cell_count"] = len(object_coords)
                        # object_dict['shape'] = generate_hash([['x' if cell != '.' else '.' for cell in row] for row in object_dict['grid']])
                        object_dict["shape"] = [
                            ["x" if cell != "." else "." for cell in row]
                            for row in object_dict["grid"]
                        ]
                    objects.append(object_dict)

        # check whether there is the color 'j'. If don't have, then do inner objects
        if not missing_color:
            # do separate empty object list for all grids in identified objects
            multicolor = False
            new_objects = []
            for obj in objects:
                visited = set()
                newgrid = obj["grid"]
                rows = len(newgrid)
                cols = len(newgrid[0])
                for row in range(rows):
                    for col in range(cols):
                        if (row, col) not in visited:
                            if newgrid[row][col] == ".":
                                object_coords = set()
                                dfs(newgrid, row, col, ".")
                                # check if we don't contain boundary coordinates
                                boundary = False
                                for x, y in object_coords:
                                    if (
                                        x == 0
                                        or x == len(newgrid) - 1
                                        or y == 0
                                        or y == len(newgrid[0]) - 1
                                    ):
                                        boundary = True
                                if boundary:
                                    continue
                                object_dict = create_object(newgrid, object_coords)
                                cur_x, cur_y = object_dict["tl"]
                                base_x, base_y = obj["tl"]

                                object_dict["tl"] = (cur_x + base_x, cur_y + base_y)
                                if more_info:
                                    object_dict["size"] = (
                                        len(object_dict["grid"]),
                                        len(object_dict["grid"][0]),
                                    )
                                    object_dict["cell_count"] = len(object_coords)
                                    # object_dict['shape'] = generate_hash([['x' if cell != '.' else '.' for cell in row] for row in object_dict['grid']])
                                    object_dict["shape"] = [
                                        ["x" if cell != "." else "." for cell in row]
                                        for row in object_dict["grid"]
                                    ]
                                new_objects.append(object_dict)

            objects.extend(new_objects)
    return objects


def get_size(grid):
    """Returns the size of the grid in 2 axis"""
    return (len(grid), len(grid[0]))
