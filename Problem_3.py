def count_islands(grid):
    if not grid:
        return 0

    rows = len(grid)
    cols = len(grid[0])
    count = 0

    def dfs(row, col):
        if row < 0 or row >= rows or col < 0 or col >= cols or grid[row][col] != '1':
            return

        grid[row][col] = '#'  # Mark the current land as visited

        # Recursively visit all neighboring lands
        dfs(row - 1, col)  # Up
        dfs(row + 1, col)  # Down
        dfs(row, col - 1)  # Left
        dfs(row, col + 1)  # Right

    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == '1':
                count += 1
                dfs(row, col)

    return count


# Example usage:
grid1 = [
    ['1', '1', '1', '1', '0'],
    ['1', '1', '0', '1', '0'],
    ['1', '1', '0', '0', '0'],
    ['0', '0', '0', '0', '0']
]
print(count_islands(grid1))  # Expected Output: 1

grid2 = [
    ['1', '1', '0', '0', '0'],
    ['1', '1', '0', '0', '0'],
    ['0', '0', '1', '0', '0'],
    ['0', '0', '0', '1', '1']
]
print(count_islands(grid2))  # Expected Output: 3