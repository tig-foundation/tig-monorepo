import { Component, signal } from '@angular/core';
import { ButtonModule } from 'primeng/button';
import { CardModule } from 'primeng/card';
import { ProgressSpinnerModule } from 'primeng/progressspinner';
import { TableModule } from 'primeng/table';
import { EditSettingsDialogComponent } from '../../components/edit-settings-dialog/edit-settings-dialog.component';
import { IconFieldModule } from 'primeng/iconfield';
import { InputIconModule } from 'primeng/inputicon';
import { TagModule } from 'primeng/tag';

@Component({
  selector: 'app-slave-manager',
  standalone: true,
  imports: [
    TableModule,
    ProgressSpinnerModule,
    ButtonModule,
    CardModule,
    EditSettingsDialogComponent,
    IconFieldModule,
    InputIconModule,
    TagModule,
  ],
  templateUrl: './slave-manager.component.html',
  styleUrl: './slave-manager.component.scss',
})
export class SlaveManagerComponent {
  challenges_data: any = signal(null);
  mining_panel_data: any = signal(null);
  slaves_data: any = signal(null);
  batches_data: any = signal(null);
  ngOnInit() {
    this.getSlavesData();
    this.getMiningPanelData();
    this.getBatchesData();
    this.getChallengesData();
  }
  getSlavesData() {
    const temp_data = [
      {
        name: 'Slave 1',
        max_concurrent_batches: 2,
      },
    ];
  }
  getMiningPanelData() {
    // Get challenges data
    const temp_data = [
      {
        name: 'Slave 1',
        max_concurrent_batches: 2,
      },
    ];
  }
  getBatchesData() {
    // Get challenges data
  }
  getChallengesData() {
    const challenges_temp_data = [
      {
        name: 'Challenge 1',
        pending: 4,
        in_progress: 2,
        average_time: 2,
        last_updated: '1 hour ago',
      },
      {
        name: 'Challenge 2',
        pending: 1,
        in_progress: 3,
        average_time: 4,
        last_updated: '1 hour ago',
      },
    ];
    this.challenges_data.set(challenges_temp_data);
  }
}
