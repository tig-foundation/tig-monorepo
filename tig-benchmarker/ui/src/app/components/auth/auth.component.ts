import { Component, inject } from '@angular/core';
import { TigApisService } from '../../services/tig-apis.service';
import { ButtonModule } from 'primeng/button';
import { InputTextModule } from 'primeng/inputtext';
import {
  FormControl,
  FormGroup,
  FormsModule,
  ReactiveFormsModule,
  Validators,
} from '@angular/forms';

@Component({
  selector: 'app-auth',
  standalone: true,
  imports: [ButtonModule, InputTextModule, ReactiveFormsModule, FormsModule],
  templateUrl: './auth.component.html',
  styleUrl: './auth.component.scss',
})
export class AuthComponent {
  tigService = inject(TigApisService);
  authFrom!: FormGroup;

  ngOnInit() {
    this.authFrom = new FormGroup({
      player_id: new FormControl('', [
        Validators.required,
        Validators.pattern('0x[a-fA-F0-9]{40}'),
      ]),
      api_key: new FormControl('', [
        Validators.required,
        Validators.minLength(32),
      ]),
    });
  }

  saveDetails() {
    if (this.authFrom.valid) {
      this.tigService.setPlayerAndAuthKey(
        this.authFrom.value.player_id,
        this.authFrom.value.api_key
      );
    }
  }
}
