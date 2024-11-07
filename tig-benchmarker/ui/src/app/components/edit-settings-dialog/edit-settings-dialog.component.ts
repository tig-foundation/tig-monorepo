import { Component, input } from '@angular/core';
import { ButtonModule } from 'primeng/button';
import { DialogModule } from 'primeng/dialog';
import { SettingsComponent } from '../../pages/settings/settings.component';

@Component({
  selector: 'app-edit-settings-dialog',
  standalone: true,
  imports: [ButtonModule, DialogModule, SettingsComponent],
  templateUrl: './edit-settings-dialog.component.html',
  styleUrl: './edit-settings-dialog.component.scss',
})
export class EditSettingsDialogComponent {
  view: any = input<string>('all');
  slave: any = input<any>(null);
  title: any = input<string>('Settings');
  edit_solution_settings = false;
}
