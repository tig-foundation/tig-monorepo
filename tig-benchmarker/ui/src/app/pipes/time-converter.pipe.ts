import { Pipe, PipeTransform } from '@angular/core';

@Pipe({
  name: 'timeConverter',
  standalone: true
})
export class TimeConverterPipe implements PipeTransform {

  transform(value: number, ...args: unknown[]): unknown {
    if(value > 0){
      const seconds = value / 1000;
      const minutes = Math.floor(seconds / 60);
      const hours = Math.floor(minutes / 60);
      const days = Math.floor(hours / 24);
      if (days > 0) {
        return days + 'd ' + hours % 24 + 'h';
      }
      else if (hours > 0) {
        return hours + 'h ' + minutes % 60 + 'm';
      }
      else if (minutes > 0) {
        return minutes + 'm ' + seconds % 60 + 's';
      }
      else {
        return seconds + 's';
      }
    }
    else {
      return 'N/A';
    }
    return null;
  }

}
