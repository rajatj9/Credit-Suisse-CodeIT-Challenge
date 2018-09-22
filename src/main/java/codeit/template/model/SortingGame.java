package codeit.template.model;

import java.util.ArrayList;
import java.util.List;

public class SortingGame {

    private final int[][] puzzle;
    private final int size;

    private final int[][] DIR = {
            {-1, 0},
            {1, 0},
            {0, -1},
            {0, 1}
    };
    private int head, tail;
    private List<int[][]> queue;
    private List<int[]> zeroPos;
    private List<Integer> rec;
    private List<Integer> xNum;

    public SortingGame(int[][] puzzle){
        this.puzzle = puzzle;
        size = puzzle.length;
        //test only
        //System.out.println("[test]size: " + puzzle.length);
    }

    private int[] locateZero() {
        int[] ret = {0, 0};
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (puzzle[i][j] == 0) {
                    ret[0] = i;
                    ret[1] = j;
                    return ret;
                }
            }
        }
        return ret;
    }

    public void solveBFS() {
        head = 0;
        tail = 0;
        queue = new ArrayList<int[][]>();
        zeroPos = new ArrayList<int[]>();
        rec = new ArrayList<Integer>();
        xNum = new ArrayList<Integer>();
        queue.add(puzzle);
        zeroPos.add(locateZero());
        rec.add(-1);
        xNum.add(0);
        while (head >= tail) {
            int[][] mat = queue.get(tail);
            int xx = zeroPos.get(tail)[0];
            int yy = zeroPos.get(tail)[1];
            //header end
            int[][] newMat;
            int newx, newy;
            int[] newZeroPos;
            int newXNum;
            for (int i = 0; i < 4 ;i++) {
                newx = xx + DIR[i][0];
                newy = yy + DIR[i][1];
                if (!(newx>=0 && newx<size && newy>=0 && newy<size)) {
                    continue;
                }
                newMat = new int[size][size];
                copyMatA2B(mat, newMat);
                newXNum = newMat[newx][newy];
                newMat[xx][yy] = newXNum;
                newMat[newx][newy] = 0;
                if (isDup(newMat)) {
                    continue;
                }
                head++;
                newZeroPos = new int[2];
                newZeroPos[0] = newx;
                newZeroPos[1] = newy;
                queue.add(newMat);
                zeroPos.add(newZeroPos);
                rec.add(tail);
                xNum.add(newXNum);
                if (isTarget(newMat)) {
                    return;
                }
            }

            //finish
            tail++;
        }

    }

    private void copyMatA2B(int[][] a, int[][] b) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                b[i][j] = a[i][j];
            }
        }
    }

    private boolean isDup(int[][] newMat) {
        for (int i = 0; i < queue.size(); i++) {
            if (isSameMat(queue.get(i), newMat)) {
                return true;
            }
        }
        return false;
    }

    private boolean isSameMat(int[][] a, int[][] b) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (a[i][j] != b[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }

    private boolean isTarget(int[][] newMat) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (i == size - 1 && j == size - 1) {
                    break;
                }
                if (newMat[i][j] != (i * size + j + 1)) {
                    return false;
                }
            }
        }
        return (newMat[size-1][size-1] == 0);
    }

    public int[] toRoute() {
        List<Integer> aRet = new ArrayList<Integer>();
        int index = head;
        while (index != 0) {
            aRet.add(xNum.get(index));
            index = rec.get(index);
        }

        int[] ret = new int[aRet.size()];
        for (int i = 0; i < aRet.size(); i++) {
            ret[i] = aRet.get(aRet.size() - 1 - i);
        }
        return ret;
    }

    //Getters & Setters
    public int[][] getPuzzle() {
        return puzzle;
    }

}
