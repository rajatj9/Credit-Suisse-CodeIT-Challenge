package codeit.template.resource;

import codeit.template.model.Square;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
public class SquareResource {

    @RequestMapping("/")
    public String hello(){
        return  "Hello world";
    }

   @RequestMapping(value = "square",method = RequestMethod.POST, produces = MediaType.APPLICATION_JSON_VALUE)
    public Integer calculateSquare(@RequestBody Map<String, String> body){
       Integer value = Integer.parseInt(body.get("input"));
       return value* value;
    }
}
