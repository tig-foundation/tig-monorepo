import { inject, Pipe, PipeTransform } from '@angular/core';
import { TigApisService } from '../services/tig-apis.service';

@Pipe({
  name: 'challengePipe',
  standalone: true
})
export class ChallengePipe implements PipeTransform {
  tigService = inject(TigApisService);
  transform(value: unknown, ...args: unknown[]): unknown {
    return this.tigService.challenges().find((a: any) => a.id === value)?.name || 'Unknown';
  }

}
