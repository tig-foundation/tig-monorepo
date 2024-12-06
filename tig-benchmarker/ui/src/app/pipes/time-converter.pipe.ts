import { Pipe, PipeTransform } from '@angular/core';

@Pipe({
  name: 'timeConverter',
  standalone: true,
})
export class TimeConverterPipe implements PipeTransform {
  transform(value: number, ...args: unknown[]): unknown {
    if (value >= 0) {
      // Convert milliseconds to seconds
      let seconds = Math.floor(value / 1000);

      // Calculate days, hours, minutes, and seconds
      const days = Math.floor(seconds / (24 * 60 * 60));
      seconds %= 24 * 60 * 60; // Remaining seconds after extracting days

      const hours = Math.floor(seconds / (60 * 60));
      seconds %= 60 * 60; // Remaining seconds after extracting hours

      const minutes = Math.floor(seconds / 60);
      seconds %= 60; // Remaining seconds after extracting minutes

      // Build the output string
      if (days > 0) {
        return `${days}d ${hours}h`;
      } else if (hours > 0) {
        return `${hours}h ${minutes}m`;
      } else if (minutes > 0) {
        return `${minutes}m ${seconds}s`;
      } else {
        return `${seconds}s`;
      }
    } else {
      return 'N/A';
    }
  }
}
