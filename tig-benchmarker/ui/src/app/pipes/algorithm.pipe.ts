import { inject, Pipe, PipeTransform } from '@angular/core';
import { TigApisService } from '../services/tig-apis.service';

@Pipe({
  name: 'algorithmPipe',
  standalone: true
})
export class AlgorithmPipe implements PipeTransform {
  tigService = inject(TigApisService);
  transform(value: unknown, ...args: unknown[]): unknown {
    return this.tigService.algorithms().find((a: any) => a.id === value)?.name || 'Unknown';
  }

}
