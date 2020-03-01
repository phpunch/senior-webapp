import React, {Component} from "react"
import Demo from "./Demo"

class DemoController extends Component {
  render() {
    return (
      <div>
        <Demo videoId="kHk5muJUwuw" prediction_filename="result_prod_c2_v2_clean.txt" />
        <Demo videoId="Y70hiPNxe4Q" prediction_filename="result_prod_clean.txt" />
      </div>
    )
  }
}

export default DemoController