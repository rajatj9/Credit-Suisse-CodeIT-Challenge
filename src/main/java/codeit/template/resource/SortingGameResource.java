package codeit.template.resource;

import codeit.template.model.SortingGame;
import net.sf.json.JSONObject;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
public class SortingGameResource {

    @RequestMapping(value = "sorting-game",method = RequestMethod.POST, produces = MediaType.APPLICATION_JSON_VALUE)
    public String calculateSquare(@RequestBody Map<String, int[][]> body){

        int[][] puzzle = body.get("puzzle");
        SortingGame sg = new SortingGame(puzzle);
        sg.solveBFS();
        int[] ans = sg.toRoute();

        JSONObject jsonRet = new JSONObject();
        jsonRet.put("result",ans);

        return jsonRet.toString();
    }

}
