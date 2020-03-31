import React, { Component } from 'react';
import ScrollMenu from 'react-horizontal-scrolling-menu';
import './Slider.css';
import { Box } from "@material-ui/core";

// list of items
// const list = [
//   { name: 'ชวน หลักภัย', time: '0'},
//   { name: 'ประยุทธ์ จันทร์โอชา', time: '0' },
//   { name: 'item3', time: '0' },
//   { name: 'item4', time: '0' },
//   { name: 'item5', time: '0' },
//   { name: 'item6', time: '0' },
//   { name: 'item7', time: '0' },
//   { name: 'item8', time: '0' },
//   { name: 'item9', time: '0' },
//   { name: 'item11', time: '0' },
//   { name: 'item12', time: '0' },
//   { name: 'item13', time: '0' },
//   { name: 'item14', time: '0' },
//   { name: 'item15', time: '0' },
//   { name: 'item16', time: '0' },
//   { name: 'item17', time: '0' },
//   { name: 'item18', time: '0' },
//   { name: 'item19', time: '0' },
// ];



// One item component
// selected prop will be passed
const MenuItem = ({ name, time, selected }) => {
  return <Box 
  borderRadius={16} 
  bgcolor="background.paper" 
  p={5}
  >
    <p>วินาทีที่ {time}</p>
    {name}
  </Box>;
};



// All items component
// Important! add unique key
export const Menu = (list, selected) =>
  list.map((el, index) => {
    const { filename, name } = el;
    const time = index * 3
    return <MenuItem name={name} time={time} key={filename} selected={selected} />;
  });


const Arrow = ({ text, className }) => {
  return (
    <div
      className={className}
    >{text}</div>
  );
};


const ArrowLeft = Arrow({ text: '<', className: 'arrow-prev' });
const ArrowRight = Arrow({ text: '>', className: 'arrow-next' });

const selected = '??';

class Slider extends Component {
  constructor(props) {
    super(props);
    // call it again if items count changes
  }
  // setCurrentTimeHandler = () => {
  //   const current_time = 3.2
  //   this.props.setCurrentTimeHandler(current_time)
  // }

  render() {
    // Create menu from items
    const menu = Menu(this.props.list, selected);

    return (
      <div>
        <ScrollMenu
          menuStyle={{display: 'flex', alignItems: 'center', userSelect: 'none', pointerEvents: 'none'}}
          data={menu}
          arrowLeft={ArrowLeft}
          arrowRight={ArrowRight}
          selected={this.props.selected}
          scrollToSelected={true}
          hideArrows={true}
        />
      </div>
    );
  }
}

export default Slider